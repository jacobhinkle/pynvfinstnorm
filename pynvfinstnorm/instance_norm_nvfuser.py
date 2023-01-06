import importlib

import torch
from torch import Tensor
from torch.nn.modules.batchnorm import _NormBase

from nvfuser._C import Fusion, FusionCache, FusionDefinition, DataType


__all__ = ["InstanceNormNVFuserFunction", "InstanceNorm3dNVFuser"]


def torch2datatype(dt: torch.dtype) -> DataType:
    """Translate between PyTorch and NVFuser element types.

    Returns `None` if the type cannot be translated.
    """
    return {
        torch.float32: DataType.Float,
        torch.float64: DataType.Double,
        torch.int32: DataType.Int32,
        torch.int64: DataType.Int,
        torch.bool: DataType.Bool,
        torch.complex64: DataType.ComplexFloat,
        torch.complex128: DataType.ComplexDouble,
    }.get(dt)


class InstanceNormNVFuserFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias,
        running_mean,
        running_var,
        use_input_stats,
        momentum,
        eps,
    ):
        channels_last = x.is_contiguous(
            memory_format=torch.channels_last
        ) or x.is_contiguous(memory_format=torch.channels_last_3d)
        if channels_last:
            order = [0] + [i for i in range(2, len(input.shape))] + [1]
            _input = x.permute(order)
        else:
            _input = x
        assert _input.is_contiguous()

        # execute fusion using Python API. Will be cached automatically
        fs = Fusion()
        fd = FusionDefinition(fs)
        with fd:
            tv_x = fd.define_tensor(x.ndim, torch2datatype(x.dtype))
            tv_weight = fd.define_tensor(weight.ndim, torch2datatype(weight.dtype))
            tv_bias = fd.define_tensor(bias.ndim, torch2datatype(bias.dtype))
            inputs = [x, weight, bias]

            if running_mean is None:
                tv_running_mean = None
                tv_running_var = None
            else:
                tv_running_mean = fd.define_tensor(
                    running_mean.shape, torch2datatype(running_mean.dtype)
                )
                tv_running_var = fd.define_tensor(
                    running_var.shape, torch2datatype(running_var.dtype)
                )
                inputs = inputs.extend([running_mean, running_var])

            # NOTE: original C++ code defined these as inputs, but unless these
            # change during runtime it may be better to treat them as constants, right?
            c_momentum = fd.define_constant(momentum)
            c_eps = fd.define_constant(eps)

            # cast inputs if necessary
            if x.dtype != torch.float32:
                tv_x = fd.ops.castOp(DataType.Float, tv_x)
            if weight.dtype != torch.float32:
                tv_weight = fd.ops.castOp(DataType.Float, tv_weight)
            if bias.dtype != torch.float32:
                tv_bias = fd.ops.castOp(DataType.Float, tv_bias)

            result = fd.ops.instance_norm(
                tv_x,
                tv_weight,
                tv_bias,
                tv_running_mean,
                tv_running_var,
                use_input_stats,
                momentum,
                eps,
                channels_last,
            )

            fd.add_output(result)
        result = fs.execute(inputs)[0]

        ctx.use_input_stats = use_input_stats
        ctx.eps = eps
        ctx.channels_last = channels_last
        # saving for backward in "explicit channels-last format"
        ctx.save_for_backward(_input, weight, running_mean, running_var, mean, invstd)
        if channels_last:
            order = [0, len(_input.shape) - 1] + [
                i for i in range(1, len(_input.shape) - 1)
            ]
            out = out.permute(order)
            if len(out.shape) == 4:
                assert out.is_contiguous(memory_format=torch.channels_last)
                assert input.is_contiguous(memory_format=torch.channels_last)
            elif len(out.shape) == 5:
                assert out.is_contiguous(memory_format=torch.channels_last_3d)
                assert input.is_contiguous(memory_format=torch.channels_last_3d)
            else:
                assert False, "unhandled channels_last format variation in forward"
        return out

    @staticmethod
    def backward(ctx, grad_output):
        global instance_norm_nvfuser_cuda
        if instance_norm_nvfuser_cuda is None:
            instance_norm_nvfuser_cuda = importlib.import_module(
                "instance_norm_nvfuser_cuda"
            )

        if ctx.channels_last:
            order = [0] + [i for i in range(2, len(grad_output.shape))] + [1]
            grad_output = grad_output.permute(order)
        # input was saved in "explicit channels-last format"
        assert ctx.saved_tensors[0].is_contiguous()
        grad_output = grad_output.contiguous()
        saved = list(ctx.saved_tensors)
        saved.insert(1, grad_output)
        running_mean = saved[3]
        running_var = saved[4]
        mean = saved[-2]
        var = saved[-1]
        grad_input, grad_weight, grad_bias = instance_norm_nvfuser_cuda.backward(
            *saved, ctx.use_input_stats, ctx.eps, ctx.channels_last
        )
        if ctx.channels_last:
            order = [0, len(grad_input.shape) - 1] + [
                i for i in range(1, len(grad_input.shape) - 1)
            ]
            grad_input = grad_input.permute(order)
            if len(grad_input.shape) == 4:
                assert grad_input.is_contiguous(memory_format=torch.channels_last)
            elif len(grad_input.shape) == 5:
                assert grad_input.is_contiguous(memory_format=torch.channels_last_3d)
            else:
                assert False, "unhandled channels_last format variation in backward"
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None


class _InstanceNormNVFuser(_NormBase):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(_InstanceNormNVFuser, self).__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )
        self.dummy = torch.empty([], device=device)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)
        # at version 1: removed running_mean and running_var when
        # track_running_stats=False (default)
        if version is None and not self.track_running_stats:
            running_stats_keys = []
            for name in ("running_mean", "running_var"):
                key = prefix + name
                if key in state_dict:
                    running_stats_keys.append(key)
            if len(running_stats_keys) > 0:
                error_msgs.append(
                    "Unexpected running stats buffer(s) {names} for {klass} "
                    "with track_running_stats=False. If state_dict is a "
                    "checkpoint saved before 0.4.0, this may be expected "
                    "because {klass} does not track running stats by default "
                    "since 0.4.0. Please remove these keys from state_dict. If "
                    "the running stats are actually needed, instead set "
                    "track_running_stats=True in {klass} to enable them. See "
                    "the documentation of {klass} for details.".format(
                        names=" and ".join(
                            '"{}"'.format(k) for k in running_stats_keys
                        ),
                        klass=self.__class__.__name__,
                    )
                )
                for key in running_stats_keys:
                    state_dict.pop(key)

        super(_InstanceNormNVFuser, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, input: Tensor) -> Tensor:
        assert input.is_cuda, "NVFuser InstanceNorm is CUDA only"
        self._check_input_dim(input)
        if self.running_mean is not None:
            out = InstanceNormNVFuserFunction.apply(
                input,
                self.weight if self.weight is not None else self.dummy,
                self.bias if self.bias is not None else self.dummy,
                self.running_mean,
                self.running_var,
                self.training or not self.track_running_stats,
                self.momentum,
                self.eps,
            )
        else:
            out = InstanceNormNVFuserFunction.apply(
                input,
                self.weight if self.weight is not None else self.dummy,
                self.bias if self.bias is not None else self.dummy,
                self.dummy,
                self.dummy,
                self.training or not self.track_running_stats,
                self.momentum,
                self.eps,
            )
        return out


class InstanceNorm3dNVFuser(_InstanceNormNVFuser):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError("expected 5D input (got {}D input)".format(input.dim()))
