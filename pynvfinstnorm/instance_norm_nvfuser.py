import importlib

import torch
from torch import Tensor
from torch.nn.modules.batchnorm import _NormBase

from nvfuser._C import DataType, Fusion, FusionCache, FusionDefinition, Scalar, Tensor

from typing import Any, Optional, Tuple


__all__ = ["InstanceNormNVFuserFunction", "InstanceNorm3dNVFuser"]


def torch2datatype(dt: torch.dtype) -> Optional[DataType]:
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


def instance_norm(
    fd: FusionDefinition,
    x: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    use_input_stats: bool,
    momentum: Scalar,
    eps: Scalar,
    channels_last: bool,
    unbiased: bool,
    extent: torch.Size,
) -> Tensor:
    """Compute instance norm layer forward for arbitrary dimensional input.

    This is a translation of `instance_norm` in NVFuser [^1] which is not
    exposed currently in the Python frontend

    [^1]: https://github.com/csarofeen/pytorch/blob/devel/third_party/nvfuser/csrc/ops/normalization.cpp#L710
    """
    assert not (
        (running_var is None) ^ (running_mean is None)
    ), "Iff running mean or var is given, the other should be"

    kBatchDim = 0
    kNumberOfDims = len(extent)
    kChannelsDim = kNumberOfDims - 1 if channels_last else 1

    x_reduction_axes = [
        axis for axis in range(kNumberOfDims) if axis not in [kBatchDim, kChannelsDim]
    ]
    x_broadcast_mask = [axis in x_reduction_axes for axis in range(kNumberOfDims)]
    B = fd.define_constant(extent[kBatchDim])

    # channels_only_broadcast_mask = [
    #    axis != kChannelsDim for axis in range(kNumberOfDims)
    # ]

    y = None
    mean = None
    invstd = None
    if use_input_stats or running_mean is None:
        # In NVFuser Python we pass correction=1 to request unbiased variance calculation
        x_var, x_mean = fd.ops.var_mean(x, x_reduction_axes, int(unbiased))
        if running_mean is not None and running_var is not None:
            one = fd.define_constant(1.0)
            rev_momentum = fd.ops.sub(one, momentum)
            # do running mean with momentum
            current_mean_hat = fd.ops.mul(x_mean, momentum)
            mean_hat = fd.ops.mul(running_mean, rev_momentum)
            new_mean_hat = fd.ops.add(mean_hat, current_mean_hat)
            new_mean_sum = fd.ops.sum(new_mean_hat, [kBatchDim])
            rB = fd.ops.reciprocal(B)
            new_mean_channels_only = fd.ops.mul(new_mean_sum, rB)
            # TODO: cast new_mean_channels_only to type of running_mean if it's half or bfloat16

            # TODO: is this a sufficient alternative to aliasOutputToInput?
            running_mean = new_mean_channels_only

            # running var calculation
            current_var_hat = fd.ops.mul(x_var, momentum)
            var_hat = fd.ops.mul(running_var, rev_momentum)
            new_var_hat = fd.ops.add(var_hat, current_var_hat)

            new_var_sum = fd.ops.sum(new_var_hat, [kBatchDim])
            new_var_channels_only = fd.ops.mul(new_mean_sum, rB)
            # TODO: Alias to input running_var
            # fd.aliasOutputToInput(new_var_channels_only, running_var)
            running_var = new_var_channels_only

        mean = x_mean
        mean_bcast = fd.ops.broadcast_in_dim(mean, extent, [kBatchDim, kChannelsDim])
        x_sub_mean = fd.ops.sub(x, mean_bcast)

        var_eps = fd.ops.add(x_var, eps)
        invstd = fd.ops.rsqrt(var_eps)
        invstd_bcast = fd.ops.broadcast_in_dim(
            invstd, extent, [kBatchDim, kChannelsDim]
        )

        y = fd.ops.mul(x_sub_mean, invstd_bcast)

    else:  # This is inference mode with running stats
        r_mean_bcasted = fd.ops.broadcast_in_dim(running_mean, extent, [kChannelsDim])
        x_sub_mean = fd.ops.sub(x, r_mean_bcasted)

        var_eps = fd.ops.add(running_var, eps)
        unbiased_invstd = fd.ops.rsqrt(var_eps)
        invstd_bcasted = fd.ops.broadcast_in_dim(
            unbiased_invstd, extent, [kChannelsDim]
        )

        # During inference, mean/invstd output are empty tensors
        # on CPU, but not on CUDA. We need to make sure we have the same
        # behavior as with eager mode on CUDA.
        mean = running_mean
        invstd = unbiased_invstd
        y = fd.ops.mul(x_sub_mean, invstd_bcast)

    if weight is not None:
        weight_bcast = fd.ops.broadcast_in_dim(weight, extent, [kChannelsDim])
        y = fd.ops.mul(y, weight_bcast)
    if bias is not None:
        bias_bcast = fd.ops.broadcast_in_dim(bias, extent, [kChannelsDim])
        y = fd.ops.add(y, bias_bcast)

    return y, mean, invstd


class InstanceNormNVFuserFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,  # contexts are actually objects of the type we are currently defining
        x: torch.Tensor,
        weight: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        running_mean: Optional[torch.Tensor],
        running_var: Optional[torch.Tensor],
        use_input_stats: bool,
        momentum: float,
        eps: float,
        unbiased: bool = False,
    ) -> torch.Tensor:
        channels_last = x.is_contiguous(
            memory_format=torch.channels_last
        ) or x.is_contiguous(memory_format=torch.channels_last_3d)
        xorig = x
        if channels_last:
            order = [0] + [i for i in range(2, len(x.shape))] + [1]
            x = x.permute(order)
        assert x.is_contiguous()

        # execute fusion using Python API. Will be cached automatically
        fs = Fusion()
        with FusionDefinition(fs) as fd:
            tv_x = fd.define_tensor(x.ndim, torch2datatype(x.dtype))
            inputs = [x]
            if weight is not None:
                assert bias is not None
                tv_weight = fd.define_tensor(weight.ndim, torch2datatype(weight.dtype))
                tv_bias = fd.define_tensor(bias.ndim, torch2datatype(bias.dtype))
                inputs.extend([weight, bias])
            else:
                tv_weight = None
                tv_bias = None

            if running_mean is None:
                tv_running_mean = None
                tv_running_var = None
            else:
                assert running_var is not None
                tv_running_mean = fd.define_tensor(
                    running_mean.ndim, torch2datatype(running_mean.dtype)
                )
                tv_running_var = fd.define_tensor(
                    running_var.ndim, torch2datatype(running_var.dtype)
                )
                inputs.extend([running_mean, running_var])
                if running_mean.dtype in [torch.half, torch.bfloat16]:
                    tv_running_mean = fd.ops.castOp(DataType.Float, tv_running_mean)
                if running_var.dtype in [torch.half, torch.bfloat16]:
                    tv_running_var = fd.ops.castOp(tv_running_var, tv_running_var)

            s_momentum = fd.define_scalar(DataType.Double)
            s_eps = fd.define_scalar(DataType.Double)
            inputs.extend([momentum, eps])

            # cast inputs if necessary
            if x is not None and x.dtype in [torch.half, torch.bfloat16]:
                tv_x = fd.ops.castOp(DataType.Float, tv_x)
            if weight is not None and weight.dtype in [torch.half, torch.bfloat16]:
                tv_weight = fd.ops.castOp(DataType.Float, tv_weight)
            if bias is not None and bias.dtype in [torch.half, torch.bfloat16]:
                tv_bias = fd.ops.castOp(DataType.Float, tv_bias)

            out, mean, invstd = instance_norm(
                fd,
                tv_x,
                tv_weight,
                tv_bias,
                tv_running_mean,
                tv_running_var,
                use_input_stats,
                s_momentum,
                s_eps,
                channels_last,
                unbiased=unbiased,
                extent=x.shape,
            )

            fd.add_output(out)
            fd.add_output(mean)
            fd.add_output(invstd)
        out, mean, invstd = fs.execute(inputs)

        ctx.use_input_stats = use_input_stats
        ctx.eps = eps
        ctx.channels_last = channels_last
        # saving for backward in "explicit channels-last format"
        ctx.save_for_backward(x, weight, running_mean, running_var, mean, invstd)
        if channels_last:
            order = [0, len(x.shape) - 1] + [i for i in range(1, len(x.shape) - 1)]
            out = out.permute(order)
            if len(out.shape) == 4:
                assert out.is_contiguous(memory_format=torch.channels_last)
                assert xorig.is_contiguous(memory_format=torch.channels_last)
            elif len(out.shape) == 5:
                assert out.is_contiguous(memory_format=torch.channels_last_3d)
                assert xorig.is_contiguous(memory_format=torch.channels_last_3d)
            else:
                assert False, "unhandled channels_last format variation in forward"
        return out

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None, None, None,]:
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
        out = InstanceNormNVFuserFunction.apply(
            input,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            self.training or not self.track_running_stats,
            self.momentum,
            self.eps,
        )
        return out


class InstanceNorm3dNVFuser(_InstanceNormNVFuser):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError("expected 5D input (got {}D input)".format(input.dim()))
