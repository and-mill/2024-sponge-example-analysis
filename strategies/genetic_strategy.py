import numpy as np
import torch

from original_sponge_examples_code.density_recorder import measure_stats

from utils.image_utils import Normalize


def generate(
    target_model,
    iterations,
    pool_size: int,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    noise_scale: float = 0.1,
    put_noise_directly: bool = False,
    preservation_size: int = 0,
    batch_size: int = 32,
    device: torch.device = torch.device("cpu"),
    print_progress: bool = False,
) -> torch.tensor:
    """
    Generate sponge example: numpy array of size (widht, height, 3).

    @param target_model: target model
    @param iterations: iterations
    @param pool_size: pool size
    @param clip_min: clip min
    @param clip_max: clip max
    @param noise_scale: noise_ cale
    @param put_noise_directly: put noise directly
    @param preservation_size: preservation size
    @param batch_size: batch size
    @param device: device
    @param print_progress: print progress
    @return: generated sponge example
    """

    # init the genetic algorithm class that holds and mutates the population
    ga = GeneticAlgorithm(
        model=target_model,
        pool_size=pool_size,
        clip_min=clip_min,
        clip_max=clip_max,
        noise_scale=noise_scale,
        put_noise_directly=put_noise_directly,
        preservation_size=preservation_size,
    )

    # -------------------------- ITERATION LOOP --------------------------
    for i in range(iterations + 1):
        # get prediction and recorded stats for whole population to get fitness
        # prepare population for density measurement (copy, rearrange, to torch)
        population = (
            torch.from_numpy(np.array(ga.population.copy())).float().permute(0, 3, 1, 2)
        )
        norm_population = Normalize()(population)
        stats_list, predictions_logits = measure_stats(
            input=norm_population,
            model=target_model,
            batch_size=batch_size,
            device=device,
        )

        # best results in this iteration
        best_id = np.argmax([s["energy_ratio"] for s in stats_list])
        best_individual = ga.population[best_id]
        # prediction
        prediction_idxs = predictions_logits.argmax(axis=1).detach().cpu().numpy()

        if print_progress:
            print(
                "{: >40} {: >40} {: >40} {: >40} {: >40} {: >40} {: >40} {: >40}".format(
                    "Iteration",
                    str(i) + " / " + str(iterations),
                    "Best post ReLU activation density in population:",
                    stats_list[best_id]["post_relu_density"],
                    "Best energy ratio in population:",
                    stats_list[best_id]["energy_ratio"],
                    "Best overall density in population:",
                    stats_list[best_id]["overall_density"],
                )
            )

        # selection + mutation based on fitness=energy_ratios
        # do not evaluate in last iteration
        if i != iterations:
            ga.selection(
                [s["energy_ratio"] for s in stats_list],
                mutation_fraction=0.01,
                best_classes=prediction_idxs,
            )

    # convert best_individual to torch.tensor with channels first, to cpu, add batch dim
    best_individual = torch.from_numpy(best_individual).permute(2, 0, 1).to('cpu').unsqueeze(0).to(torch.float32)

    return best_individual  # best_individual from last iteration
