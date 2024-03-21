import torch

from original_sponge_examples_code.density_recorder import DensityRecorder, HardwareModel

from utils.image_utils import (
    rand_images,
    Normalize,
    UnNormalize,
    clamp_image,
)
from utils.imagenet_idx_to_class_names import idx_to_class_names


def generate(
    target_model: torch.nn.Module,
    epochs: int,
    lr: float = 1.0,
    device: torch.device = torch.device("cpu"),
    print_progress: bool = False,
) -> torch.tensor:
    """
    Generate sponge example: numpy array of size (widht, height, 3).

    @param target_model: target model
    @param epochs: epochs
    @param lr: learning rate
    @param device: device
    @param print_progress: print progress
    @return: generated sponge example
    """

    # weights and biases are no leaf nodes in computational graph
    for p in target_model.parameters():
        p.requires_grad = False

    # prepare transformations
    normalize = Normalize()
    un_normalize = UnNormalize()

    # initialize input
    input = rand_images(amount=1).to(device)
    input = normalize(input)

    # initialize optimizer
    img = torch.nn.Parameter(input, requires_grad=True)
    params = [img]
    optimizer = torch.optim.LBFGS(params, lr=lr)

    # init density recorder and hardware model
    hardware_w_optim = HardwareModel(optim=True, device=device)
    hardware_wo_optim = HardwareModel(optim=False, device=device)
    with DensityRecorder(
        model=target_model, batch_size=1, records_l2_norm=True, device=device
    ) as density_recorder:
        # -------------------------- EPOCH LOOP --------------------------
        for i in range(epochs + 1):
            # ------------------------------ RETRIEVE STATS ------------------------------
            with torch.no_grad():

                # inference + prediction
                density_recorder.__reset__()
                predictions_logits = target_model(params[0])
                prediction_idxs = predictions_logits.argmax(axis=1)
                predicted_class_ids = list(set(prediction_idxs.detach().cpu().numpy()))
                predicted_class_names = [
                    idx_to_class_names[i] for i in predicted_class_ids
                ]
                predicted_class_probs = (
                    torch.nn.functional.softmax(predictions_logits, dim=1)
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )
                predictions_logits = predictions_logits.detach().cpu().numpy().tolist()

                # get recorded stats
                stats_list = density_recorder.get_results()

            # This is only one iteration - gradient sponges work on a single image - all lists are single elements wrapped in extra dimension
            for (
                predicted_class_id,
                predicted_class_name,
                predicted_logits,
                predicted_class_probabilities,
                stats,
            ) in zip(
                predicted_class_ids,
                predicted_class_names,
                predictions_logits,
                predicted_class_probs,
                stats_list,
            ):
                if print_progress:
                    print(
                        "{: >40} {: >40} {: >40} {: >40} {: >40} {: >40} {: >40} {: >40}".format(
                            "Epoch",
                            str(i) + " / " + str(epochs),
                            "Energy ratio:",
                            stats["energy_ratio"],
                            "Post relu density:",
                            stats["post_relu_density"],
                            "Overall density:",
                            stats["overall_density"],
                        )
                    )

            # ------------------------------ optimizer step, but not in last epoch ------------------------------
            if i != epochs + 1:
                # ------------------------------ LBFGS ------------------------------
                def closure():
                    """
                    LBFGS uses a closure function.
                    """
                    optimizer.zero_grad()

                    density_recorder.__reset__()
                    pred = target_model(params[0])
                    density_recorder.l_norm_activation.backward(retain_graph=True)

                    return density_recorder.l_norm_activation.item()

                optimizer.step(closure)

                # pre processing for next iteration
                optimizer.param_groups[0]["params"][0].data = clamp_image(
                    image=optimizer.param_groups[0]["params"][0].data,
                    clamp_min=0,
                    clamp_max=1,
                    normalized=True,
                )

        # extract image from optimizer, convert to single image, unnormalized, with no batch dim, to cpu, add batch dim
        result = UnNormalize()(optimizer.param_groups[0]["params"][0].data.detach()).squeeze(0).to('cpu').unsqueeze(0)

        return result
