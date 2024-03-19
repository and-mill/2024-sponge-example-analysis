"""
This Code is mostly copied from [sponge-examples](https://github.com/iliaishacked/sponge_examples/blob/main/energy_estimator/analyse.py) and then modified to integrate into our project.
Please see the original license in the `LICENSE-original` file.
Please cite their work correctly:
@inproceedings{shumailov2020sponge,
      title={Sponge Examples: Energy-Latency Attacks on Neural Networks}, 
      author={Ilia Shumailov and Yiren Zhao and Daniel Bates and Nicolas Papernot and Robert Mullins and Ross Anderson},
      year={2021},
      booktitle={6th IEEE European Symposium on Security and Privacy (EuroS\&P)},
}
"""

import torch

from utils.image_utils import batch_tensor, Normalize

BIT_WIDTH_TO_MIN_VALUE = {
    32: 2**-126,
    16: 2**-30,
    8: 2**-14,
}
# This is necessary for determining zero activations
# We assume 32 bit precision for floats.
MIN_VALUE = BIT_WIDTH_TO_MIN_VALUE[32]


def l2_norm(x: torch.tensor) -> torch.tensor:
    return x.square().sum().sqrt()


def nonzero_func_batch(x: torch.tensor) -> torch.tensor:
    """
    Function for counting nonzero values in a feature map that IS batch-dependant,
    i.e. ReLU in and out

    :param x: feature map tensor
    :return: nonzero tensor elements in x as 1D float tensor with <x.shape[0]> elements (batch dim of x)
    """
    res = x.abs() > MIN_VALUE
    res = res.flatten(start_dim=1)
    res = res.sum(dim=1).float()
    return res


def nonzero_func(x: torch.tensor) -> torch.tensor:
    """
    Function for counting nonzero values in a feature map that IS NOT batch-dependant,
    i.e. nonzero_func(module.running_mean)

    :param x: feature map tensor
    :return: nonzero tensor elements in x as float
    """
    return float(len((x.abs() > MIN_VALUE).nonzero()))


class HardwareModel:
    """
    Just a simple container for memory and computation costs.

    Default energy figures taken from here (45nm):
    https://ieeexplore.ieee.org/document/6757323

    Assuming all operations are 32 bit floating point (others are available).
    """

    def __init__(self, optim: bool = True, device: torch.device = torch.device("cpu")):
        # Cost of a single memory access in pJ.
        # ASSUMPTION: DRAM costs dominate (on-chip caches are negligible).
        # ASSUMPTION: reads and writes cost the same.
        self.memory_cost = torch.tensor(1950.0, dtype=torch.float, device=device)

        # Cost of a single computation (e.g. multiply) in pJ.
        # ASSUMPTION: all computations cost the same amount.
        self.compute_cost = torch.tensor(3.7, dtype=torch.float, device=device)

        # Is the hardware able to optimise memory access for sparse data?
        # ASSUMPTION: there is no overhead to (de)compression.
        self.compress_sparse_weights = optim
        self.compress_sparse_activations = optim

        # Is the hardware able to skip computations when one input is zero?
        # ASSUMPTION: zeros are uniformly distributed throughout the data.
        self.compute_skip_zero_weights = optim
        self.compute_skip_zero_activations = optim


class DensityRecorder:
    """
    A context for capturing activations inside a Pytorch model. Use it like this:

    with StatsRecorder(model) as sr:
        ...
        energy_consumed = get_energy_estimate(HardwareModel)
       ...

    If l_norm_fn is not None, it also captures a differentiable accumulation results
    for sponge attacks using gradient descend.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        batch_size: int,
        records_l2_norm: bool = False,
        device: torch.device = torch.device("cpu"),
    ):
        """
        @param model: model to record on
        @param batch_size: batch_size of processed inputs
        @param records_l2_norm: sum up L2 norm of all feature maps for Sponge-LBFGS
        @param device: device
        """

        assert batch_size > 0, "batch size must be greater than 0"
        # summing up L2 norm of all feature maps for Sponge-LBFGS. Only works with batch_size=1 right now.
        assert (
            not records_l2_norm or batch_size == 1
        ), "recording L2 norm is set True. Summing up L2 norm of all feature maps does not work with batch sizes > 1."

        self.model = model
        self.batch_size = batch_size
        self.records_l2_norm = records_l2_norm
        self.device = device

        # init the stats
        self.__reset__()

        # prepare hooks
        self.hooks = []

    def __reset__(self):
        """
        Reset attributes, but model and hooks stay
        """
        self.total_input_activations = torch.zeros(self.batch_size, dtype=torch.float, device=self.device)
        self.non_zero_input_activations = torch.zeros(self.batch_size, dtype=torch.float, device=self.device)
        self.total_output_activations = torch.zeros(self.batch_size, dtype=torch.float, device=self.device)
        self.non_zero_output_activations = torch.zeros(self.batch_size, dtype=torch.float, device=self.device)
        self.total_parameters = torch.zeros(self.batch_size, dtype=torch.float, device=self.device)
        self.non_zero_parameters = torch.zeros(self.batch_size, dtype=torch.float, device=self.device)
        self.computations = torch.zeros(self.batch_size, dtype=torch.float, device=self.device)
        self.total_post_relu_activations = torch.zeros(self.batch_size, dtype=torch.float, device=self.device)
        self.non_zero_post_relu_activations = torch.zeros(self.batch_size, dtype=torch.float, device=self.device)
        self.l_norm_activation = torch.zeros(self.batch_size, dtype=torch.float, device=self.device)

    def get_post_relu_density(self) -> torch.Tensor:
        """
        Post-ReLU activation density
        :return: post relu density
        """
        return self.non_zero_post_relu_activations / self.total_post_relu_activations

    def get_energy_estimate(self, hw: HardwareModel) -> torch.tensor:
        """
        Estimate the energy consumption in picojoules of a given computation on
        given hardware.

        ASSUMPTIONS:
        * Weights are read from DRAM exactly once.
        * Input activations are read from DRAM exactly once.
        * Output activations are written to DRAM exactly once.

        @param hw: a HardwareModel containing details of the processor.

        @return: cost
        """

        # memory cost
        memory_cost = torch.zeros(self.batch_size, device=self.device)
        if hw.compress_sparse_weights:
            memory_cost += hw.memory_cost * self.non_zero_parameters
        else:
            memory_cost += hw.memory_cost * self.total_parameters

        if hw.compress_sparse_activations:
            memory_cost += hw.memory_cost * (
                self.non_zero_input_activations + self.non_zero_output_activations
            )
        else:
            memory_cost += hw.memory_cost * (
                self.total_input_activations + self.total_output_activations
            )

        # computation cost
        compute_fraction = torch.ones(self.batch_size, device=self.device)
        if hw.compute_skip_zero_weights:
            compute_fraction *= self.non_zero_parameters / self.total_parameters
        if hw.compute_skip_zero_activations:
            compute_fraction *= (
                self.non_zero_input_activations / self.total_input_activations
            )

        return memory_cost + (compute_fraction * self.computations * hw.compute_cost)

    def get_energy_ratio(self) -> torch.tensor:
        """
        Energy ratio as used by Shumailov et al.
        :return: energy ratio
        """
        return self.get_energy_estimate(
            HardwareModel(optim=True, device=self.device)
        ) / self.get_energy_estimate(HardwareModel(optim=False, device=self.device))

    def get_overall_density(self) -> torch.tensor:
        """
        Overall density as used by Shumailov et al.

        :return: overall density
        """
        overall_density_total = (
            self.total_input_activations
            + self.total_output_activations
            + self.total_parameters
        )
        overall_density_non_zeros = (
            self.non_zero_input_activations
            + self.non_zero_output_activations
            + self.non_zero_parameters
        )
        return overall_density_non_zeros / overall_density_total

    def get_results(self) -> torch.tensor:
        """
        Get post-relu-density, energy estimate, and overall density as list of dicts
        :return: list of results dict of str: float
        """
        return [
            {"post_relu_density": r.item(), "energy_ratio": e.item(), "overall_density": o.item()}
            for r, e, o in zip(
                self.get_post_relu_density(),
                self.get_energy_ratio(),
                self.get_overall_density(),
            )
        ]

    def __enter__(self):
        self.__add_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__remove_hooks()
        self.__reset__()

    def __add_hooks(self):
        """
        Prepare a model for analysis by adding hooks.

        Intercept computation in each leaf node of the network, and collect data
        on the amount of data accessed and computation performed.

        ASSUMPTION: nothing significant happens in modules which contain other
        modules. Only leaf modules are analysed.
        """
        leaf_nodes = [
            module
            for module in self.model.modules()
            if len(list(module.children())) == 0
        ]

        stat_fn = record_stats_batch_wise(self)
        for module in leaf_nodes:
            hook = module.register_forward_hook(stat_fn)
            self.hooks.append(hook)

    def __remove_hooks(self):
        """
        Remove hooks from a model.
        """
        for hook in self.hooks:
            hook.remove()


# ---------------------------------------------------------------------------------------------------------------
# Helper function that is hooked into a PyTorch model
def record_stats_batch_wise(recorder: DensityRecorder) -> callable:
    """
    Create a forward hook function which will record information about a single layer's
    execution.

    Supports batches and seperates data for each input in batch.
    There are two "batch sensitive" things here
    1. instead of
        - tensor.numel()
        - in_data[0].numel()
        - out_data[0].numel()
        do var[0, :].numel() to return a tensor that splits numel to batchsize
    2. instead of nonzero_func(...) do nonzero_func_batch(...). We can't use nonzero_func_batch(...)
       in all cases because nonzero_func(...) is sometimes applied to tensors with few dimensions and the
       code will not deal with it properly.
    all lines where we have to apply 1 or 2 are marked with "batch sensitive" in the code

    For all module parameters/buffers, in_data and out_data, record:
    * Number of values
    * Number of non-zeros
    Also estimate amount of computation (depends on layer type).

    @param recorder: a StatsRecorder to store results in

    @return: forward hook function
    """

    def hook_fn(
        module: torch.nn.Module,
        in_data: torch.Tensor,
        out_data: torch.Tensor,
    ):
        """
        Receives a DensityRecorder from outer scope instance to fill up during forward pass

        @param module: module
        @param in_data: in_data
        @param out_data: out_data
        """

        # Activations are sometimes Tensors, and sometimes tuples of Tensors.
        # Ensure we're always dealing with tuples.
        if isinstance(in_data, torch.Tensor):
            in_data = (in_data,)
        if isinstance(out_data, torch.Tensor):
            out_data = (out_data,)
        # in_data and out_data are now a 1-tuple of tensors with shape (batch_size, d0, d1, d2)

        # -------------------------- collect memory info --------------------------
        for tensor in in_data:
            # collect detailed information
            recorder.total_input_activations += tensor[0, :].numel()  # batch sensitive
            recorder.non_zero_input_activations += nonzero_func_batch(
                tensor
            )  # batch sensitive

        for tensor in out_data:
            recorder.total_output_activations += tensor[0, :].numel()  # batch sensitive
            recorder.non_zero_output_activations += nonzero_func_batch(
                tensor
            )  # batch sensitive

            # for post-ReLu density
            if "RELU" in type(module).__name__.upper():
                recorder.total_post_relu_activations += tensor[
                    0, :
                ].numel()  # batch sensitive
                recorder.non_zero_post_relu_activations += nonzero_func_batch(
                    tensor
                )  # batch sensitive

                if l2_norm is not None:
                    recorder.l_norm_activation += l2_norm(
                        tensor[0].flatten()
                    )  # batch sensitive

        for tensor in module.buffers():
            recorder.total_parameters += tensor.numel()
            recorder.non_zero_parameters += nonzero_func(tensor)

        for tensor in module.parameters():
            recorder.total_parameters += tensor.numel()
            recorder.non_zero_parameters += nonzero_func(tensor)

        # -------------------------- collect computations --------------------------

        if isinstance(module, torch.nn.AdaptiveAvgPool2d):
            # One computation per input pixel - window size is chosen adaptively
            # and windows never overlap (?).
            assert len(in_data) == 1
            input_size = in_data[0][0, :].numel()  # batch sensitive
            recorder.computations += input_size

        elif isinstance(module, torch.nn.AvgPool2d) or isinstance(
            module, torch.nn.MaxPool2d
        ):
            # Each output pixel requires computations on a 2D window of input.
            if type(module.kernel_size) is int:
                # Kernel size here can be either a single int for square kernel
                # or a tuple (see
                # https://pytorch.org/docs/stable/nn.html#torch.nn.MaxPool2d )
                window_size = module.kernel_size**2
            else:
                window_size = module.kernel_size[0] * module.kernel_size[1]

            # Not sure which output tensor to use if there are multiple of them.
            assert len(out_data) == 1
            output_size = out_data[0][0, :].numel()  # batch sensitive
            recorder.computations += output_size * window_size

        elif isinstance(module, torch.nn.Conv2d):
            # Each output pixel requires computations on a 3D window of input.
            # Not sure which input tensor to use if there are multiple of them.
            assert len(in_data) == 1
            _, channels, _, _ = in_data[0].size()  # here no batch, just want channels
            window_size = module.kernel_size[0] * module.kernel_size[1] * channels

            # Not sure which output tensor to use if there are multiple of them.
            assert len(out_data) == 1
            output_size = out_data[0][0, :].numel()  # batch sensitive

            recorder.computations += output_size * window_size

        elif isinstance(module, torch.nn.Linear):
            # One computation per weight, for each batch element.

            # Not sure which input tensor to use if there are multiple of them.
            assert len(in_data) == 1
            batch = in_data[0][0, :].numel() / in_data[0].shape[-1]  # batch sensitive

            recorder.computations += module.weight.numel() * batch

        elif isinstance(module, torch.nn.modules.batchnorm.BatchNorm2d):
            # Accesses to E[x] and Var[x] (all channel size)

            recorder.total_parameters += 2 * module.num_features
            recorder.non_zero_parameters += nonzero_func(
                module.running_mean
            ) + nonzero_func(module.running_var)

            # (x-running_mean)/running variance
            # multiply by gamma and beta addition
            recorder.computations += 4 * in_data[0][0, :].numel()  # batch sensitive

        elif isinstance(
            module,
            (
                torch.nn.modules.activation.ReLU,
                torch.nn.modules.activation.ReLU6,
                torch.nn.LayerNorm,
                torch.nn.Dropout2d,
                torch.nn.modules.linear.Identity,
                torch.nn.modules.flatten.Flatten,
            ),
        ):
            pass

        else:
            print("Unsupported module type for energy analysis: ", type(module))

    return lambda *x: hook_fn(*x)


def measure_stats(
    input: torch.Tensor,
    model: torch.nn.Module,
    batch_size: int,
    device: torch.device = torch.device("cpu"),
) -> tuple[list, torch.tensor]:
    """
    High-level function for measuring all stats, as well as predictions, on multiple data (has batch dimension) in one go.
    Assumption: input is already normalized.

    @param input: input, already normalized
    @param model: model
    @param batch_size: batch_size to run on device. This is NOT the batch size of input
    @param device: device
    @return predictions_logits, stats_list
    """

    # input has shape (self.pool_size, 3, input_shape, input_shape)
    batches = batch_tensor(input, batch_size)

    # -------------------------- process mini batches --------------------------
    stats_list = None
    predictions_logits = None
    for batch in batches:
        this_batch_size = batch.size(dim=0)
        batch = batch.to(device)

        # inference + prediction
        with torch.no_grad(), DensityRecorder(
            model, batch_size=this_batch_size, device=device
        ) as stats_recorder:

            # forward pass
            stats_recorder.__reset__()
            predictions_logits_batch = model(batch)

            # collect stats from different batches
            stats_list_batch = (
                stats_recorder.get_results()
            )  # list of dicts with {'post_relu_density': ...}
            stats_list = (
                stats_list_batch
                if stats_list is None
                else stats_list + stats_list_batch
            )
            # collect prediction logits from different batches
            predictions_logits = (
                predictions_logits_batch
                if predictions_logits is None
                else torch.cat([predictions_logits, predictions_logits_batch], dim=0)
            )

    assert len(stats_list) == len(predictions_logits)

    return stats_list, predictions_logits


def measure_post_relu_density(
    input: torch.Tensor,
    model: torch.nn.Module,
    batch_size: int,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.tensor]:
    """
    High-level function for measuring a single stat on multiple data (has batch dimension).
    Assumption: input is already normalized.

    @param input: input, already normalized
    @param model: model
    @param batch_size: batch_size to run on device. This is NOT the batch size of input
    @param device: device
    @return predictions_logits, stats_list
    """

    # input has shape (self.pool_size, 3, input_shape, input_shape)
    batches = batch_tensor(input, batch_size)

    # -------------------------- process mini batches --------------------------
    stats_tensor = None
    for batch in batches:
        this_batch_size = batch.size(dim=0)
        batch = batch.to(device)

        # inference + prediction
        with torch.no_grad(), DensityRecorder(
            model, batch_size=this_batch_size, device=device
        ) as stats_recorder:

            # forward pass
            stats_recorder.__reset__()
            _ = model(batch)

            # collect stats from different batches
            stats_tensor_batch = (
                stats_recorder.get_post_relu_density()
            )
            stats_tensor = (
                stats_tensor_batch
                if stats_tensor is None
                else torch.cat([stats_tensor, stats_tensor_batch], dim=0)
            )

    return stats_tensor
