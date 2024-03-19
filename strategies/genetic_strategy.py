import numpy as np
import torch

from density_recorder.density_recorder import measure_stats

from utils.imagenet_idx_to_class_names import idx_to_class_names

from utils.image_utils import rand_images, Normalize


class GeneticAlgorithm:
    def __init__(
        self,
        model: torch.nn.Module,
        pool_size: int,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        noise_scale: float = 0.1,
        put_noise_directly: bool = False,
        preservation_size: int = 0,
    ):
        """
        @param model: model
        @param pool_size: pool_size
        @param clip_min: clip_min
        @param clip_max: clip_max
        @param noise_scale: noise_scale
        @param put_noise_directly: put_noise_directly
        @param preservation_size: int = 0
        """
        self.model = model
        self.pool_size = pool_size
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.noise_scale = noise_scale
        self.put_noise_directly = put_noise_directly
        self.preservation_size = preservation_size

        # init population
        self.population = rand_images(amount=pool_size)
        self.population = self.population.permute(0, 2, 3, 1).detach().cpu().numpy()
        # The original code worked with numpies and channels first, so I keep it that way

    def __scaled_random_noise(self, shape: np.array, scale: float, dtype) -> np.array:
        """
        Random noise of shape in range [self.clip_min...self.clip_max] * scale as dtype
        """
        noise = np.random.rand(*shape) * scale
        noise = self.__clip(noise)
        return noise.astype(dtype)

    def __random_pick(
        self, pool: np.array, axis: int, p: np.array = None, normalize: bool = True
    ) -> np.array:
        """
        Pick randomly along axis based on probabilities

        @param pool: pool to pick from
        @param p: probabilities
        @param normalize: whether to normalize p

        @return: picks from pool
        """
        assert len(pool) == len(p)
        # normalize probabilities
        if normalize:
            p = p / p.sum()
        # pick best indexes and collect from pool
        return pool[np.random.choice(pool.shape[axis], p=p), :]

    def selection(
        self,
        fitness_scores: np.array,
        mutation_fraction: float = None,
        best_classes: np.array = None,
    ):
        """
        Do selection, cross-over, mutation.
        self.population is modified by this.

        @param fitness_scores: fitness_scores corresponding to each pop in population
        @param mutation_fraction: float in [0...1] to enable partial mutation of inputs
        @param best_classes: 1D-array of size len(population) containing the class index for each pop
                             to enable optimization where classes are provided
        """

        # sort population and scores
        scores_sorted, population_sorted = zip(
            *[
                [s, p]
                for s, p in sorted(
                    zip(fitness_scores, self.population), key=lambda x: x[0]
                )
            ]
        )
        scores_sorted = list(scores_sorted)
        population_sorted = list(population_sorted)
        percentile10 = int(len(population_sorted) * 0.1)

        # parents, top_scores, and population will now be lists of arrays

        # top 10% percent are new parents
        parents = population_sorted[-percentile10:]
        top_scores = scores_sorted[-percentile10:]

        # init new population container
        population = parents

        # best class optimization. Collect up to <preservation_size> best classes for the next round.
        # if preservation_size ==  0 -> skip
        # if preservation_size == -1 -> unlimited
        if best_classes is not None and self.preservation_size != 0:
            # collect index of fitest pop for each class index into
            # dict(class_index: (score, pop_index))
            score_popidx_for_classidx = {
                class_id: (score, pop_id)
                for pop_id, (score, class_id) in enumerate(
                    zip(fitness_scores, best_classes)
                )
            }
            preserved_classes = list(score_popidx_for_classidx.keys())
            print(
                "{: >30} {: >30} {: >30} {: >50} {: >30} {: >50}".format(
                    "Class variety",
                    str(len(preserved_classes)),
                    "Preserved Class IDs",
                    str(sorted(preserved_classes)),
                    "Preserved Class Names",
                    str([idx_to_class_names[e] for e in sorted(preserved_classes)]),
                )
            )

            # sort dict descending
            score_popidx_for_classidx = {
                class_index: (score, pop_id)
                for class_index, (score, pop_id) in sorted(
                    score_popidx_for_classidx.items(), key=lambda x: -x[1][0]
                )
            }

            # append pop_indexes for each class to parents and population. Only take <preservation_size> times
            for class_index, (score, pop_id) in list(score_popidx_for_classidx.items())[
                0 : self.preservation_size
            ]:
                parents.append(self.population[pop_id])
                population.append(self.population[pop_id])
                top_scores.append(score)

        # parents, top_scores will now be arrays
        parents = np.array(parents)
        top_scores = np.array(top_scores)

        # collect mutated pops until pool is full
        while len(population) < self.pool_size:
            # cross over
            parent_a = self.__random_pick(parents, axis=0, p=top_scores, normalize=True)
            parent_b = self.__random_pick(parents, axis=0, p=top_scores, normalize=True)
            offspring = [  # list of arrays
                self.__crossover(parent_a, parent_b),
                # self.__crossover(parent_b, parent_a)  # TODO: see if this had some deeper sense
            ]
            # mutate and append to population
            for offspr in offspring:
                offspr = self.__mutate(offspr, mutation_fraction=mutation_fraction)
                population.append(offspr)

        np.random.shuffle(population)
        self.population = population

    def __crossover(self, a: np.array, b: np.array, concat: bool = False) -> np.array:
        """
        Crossover in two modes
        1. concat mode: glue both at middle point
        2. masked mode: mask is random number in [0...1] for each pixel channel

        @param a: parent a
        @param b: parent b
        @param concat: concat mode if ture, else masked

        @return: offspring
        """
        # flatten both parents and find middle point
        a_shape = a.shape
        flat_a, flat_b = a.flatten(), b.flatten()
        mid_point = len(flat_a) // 2

        # concat mode
        if concat:
            left = flat_a[:mid_point]
            right = flat_b[mid_point:]
            offspring = np.concatenate((left, right), axis=0)
        # masked mode
        else:
            masked = np.random.rand(*flat_a.shape)
            offspring = flat_a * masked + flat_b * (1 - masked)

        # unflatten and return
        return offspring.reshape(a_shape)

    def __mutate(self, a: np.array, mutation_fraction: float = None) -> np.array:
        """
        Mutate input

        @param a: pop
        @param mutation_fraction: fraction of indices to mutate

        @return: mutated pop
        """

        # save shape for later
        a_shape = a.shape

        # only mutate a fraction of indices, defined by perc
        if mutation_fraction is not None:
            # get int(a * perc) random indices in range [0...a.size] (a.size = len(a.flatten()))
            # so if perc is 0.01, there are only 1/100 indices chosen
            a = a.flatten()
            indx = (
                np.random.random(size=int(mutation_fraction * a.size)) * a.size
            ).astype(int)
            noise = self.__scaled_random_noise(
                shape=a.shape, scale=self.noise_scale, dtype=a.dtype
            )

            # The original code did not do this. It put the noise directly into the pop. This was weird.
            if not self.put_noise_directly:
                centered_noise = (
                    noise - self.noise_scale * float(self.clip_max - self.clip_min) / 2
                )
                added = a + centered_noise  # added this to try out
                clipped = self.__clip(added)  # added this to try out
                noise = clipped

            np.put(a, indx, noise)
            result = a.reshape(a_shape)

        else:
            # get noise, center it at zero, subtract from input, clip, return
            noise = self.__scaled_random_noise(
                shape=a.shape, scale=self.noise_scale, dtype=a.dtype
            )

            # The original code did not do this. It put the noise directly into the pop. This was weird.
            if not self.put_noise_directly:
                centered_noise = (
                    noise - self.noise_scale * float(self.clip_max - self.clip_min) / 2
                )
                added = a + centered_noise  # added this to try out
                clipped = self.__clip(added)  # added this to try out
                result = clipped

        return result

    def __clip(self, x: np.array) -> np.array:
        """
        Clip a numpy array to between self.clip_min and self.clip_max

        @param x: array to clip

        @return: clipped
        """
        return np.clip(x, self.clip_min, self.clip_max)


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
