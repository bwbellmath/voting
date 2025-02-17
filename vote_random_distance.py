import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import matplotlib.pyplot as plt

class VoterModel(pl.LightningModule):
    """
    A simple PyTorch Lightning module that encapsulates methods for:
    1. Generating voters with specified orientation preferences and variance
    2. Converting those voters' (score) preferences into:
       - Single-preference (plurality) votes
       - Ranked-order votes
       - Approval votes (based on > (mean + std) for each voter)
    """

    def __init__(self):
        super().__init__()

    def generate_voters(self, orientations: dict, num_voters: int, variance: float) -> torch.Tensor:
        """
        Generate a tensor of voter preferences.

        Args:
            orientations (dict): A dictionary of orientation_name -> fraction desired.
                                 e.g., {"red": 0.25, "yellow": 0.33, "green": 0.33, "blue": 0.09}
                                 The values should sum (approximately) to 1.
            num_voters (int): Number of voters to generate.
            variance (float): Controls how much individual voters deviate from the 
                              baseline "perfect preference" for their chosen orientation.
                              0.0 means no variation within an orientation group; 
                              higher means more spread.

        Returns:
            voters (torch.Tensor): Shape [num_voters, num_orientations].
                                   Each row sums to 1.0, representing the scores a voter 
                                   assigns to each orientation.
        """
        orientation_names = list(orientations.keys())
        orientation_fractions = list(orientations.values())
        num_orientations = len(orientation_names)

        # Convert fractions to exact counts (rounding) for how many voters prefer each orientation
        counts = [round(frac * num_voters) for frac in orientation_fractions]
        total_assigned = sum(counts)
        # Adjust if rounding error occurs
        if total_assigned < num_voters:
            idx = torch.argmax(torch.tensor(orientation_fractions)).item()
            counts[idx] += (num_voters - total_assigned)
        elif total_assigned > num_voters:
            idx = torch.argmax(torch.tensor(orientation_fractions)).item()
            counts[idx] -= (total_assigned - num_voters)

        # Generate voters for each orientation
        all_voters = []
        for i, c in enumerate(counts):
            if c == 0:
                continue

            # Baseline vector: orientation i is "1.0" and others "0.0"
            baseline = torch.zeros(num_orientations)
            baseline[i] = 1.0

            # Expand baseline to (c, num_orientations)
            baseline_batch = baseline.unsqueeze(0).repeat(c, 1)

            if variance > 0:
                # Add random noise, clamp >= 0, renormalize
                noise = torch.randn(c, num_orientations) * variance
                baseline_batch = baseline_batch + noise
                baseline_batch = torch.clamp(baseline_batch, min=0.0)
                # Renormalize each row to sum to 1
                row_sums = baseline_batch.sum(dim=1, keepdim=True)
                row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
                baseline_batch = baseline_batch / row_sums

            all_voters.append(baseline_batch)

        voters = torch.cat(all_voters, dim=0)  # [num_voters, num_orientations]

        # Shuffle so we don't keep them grouped by orientation
        perm = torch.randperm(voters.size(0))
        voters = voters[perm]

        return voters

    def vote_single_preference(self, voters: torch.Tensor) -> torch.Tensor:
        """
        Convert a [num_voters, num_orientations] score matrix into 
        single-preference (plurality) votes.

        Each voter's row is converted to one-hot, where 1 is at the index
        of that row's argmax, and 0 otherwise.

        Example:
            [0.2, 0.4, 0.3, 0.1] -> [0, 1, 0, 0]
        """
        max_indices = torch.argmax(voters, dim=1)
        return F.one_hot(max_indices, num_classes=voters.shape[1]).float()

    def vote_ranked(self, voters: torch.Tensor) -> torch.Tensor:
        """
        Convert a [num_voters, num_orientations] score matrix into 
        ranked votes from 1..M.

        For each voter, the highest score gets rank=1, second-highest score gets rank=2, etc.
        Ties are broken by the ordering from argsort (stable for identical elements).

        Example:
            If a single row is [0.2, 0.4, 0.3, 0.1],
            argsort(descending=True) might give indices [1, 2, 0, 3].
            Then the rank for index 1 is 1, index 2 is 2, index 0 is 3, index 3 is 4.

            So the returned row is [3, 1, 2, 4].
        """
        sorted_indices = torch.argsort(voters, dim=1, descending=True)
        ranks = torch.argsort(sorted_indices, dim=1) + 1
        return ranks

    def vote_approval(self, voters: torch.Tensor) -> torch.Tensor:
        """
        Approval voting where each voter approves all orientations whose
        score is > (mean + std) for that voter.

        If std=0 for a voter, that row won't approve any orientation.
        """
        row_means = voters.mean(dim=1, keepdim=True)  # shape [N, 1]
        row_stds = voters.std(dim=1, keepdim=True)    # shape [N, 1]
        threshold = row_means + row_stds
        return (voters > threshold).float()


def demo():
    """
    Demo:
      1) Create a config of 10 orientations (each 0.1)
      2) Generate 100 voters with variance=0.1
      3) Get single-preference, ranked, and approval votes
      4) Sort all the data by which orientation each voter prefers (argmax),
         then reorder everything in that order.
      5) Plot a combined heatmap
    """
    model = VoterModel()

    # 1) Example orientation config for 10 orientations
    orientations = {f"orientation_{i+1}": 0.1 for i in range(10)}

    num_voters = 100
    variance = 0.1

    # Generate voters
    voters = model.generate_voters(orientations, num_voters, variance)

    # Derive different voting methods
    sp = model.vote_single_preference(voters)
    rk = model.vote_ranked(voters)
    ap = model.vote_approval(voters)

    # 4) Sort by argmax of each voter's row
    max_indices = torch.argmax(voters, dim=1)  # shape [num_voters]
    sorted_indices = torch.argsort(max_indices)  # ascending sort by orientation index
    # If you'd like to group voters who prefer orientation 0, then orientation 1, etc.
    # This will place all orientation-0-lovers first, then orientation-1, etc.
    # Ties do not matter here since argmax is a single index.

    # Reorder
    voters_sorted = voters[sorted_indices]
    sp_sorted = sp[sorted_indices]
    rk_sorted = rk[sorted_indices]
    ap_sorted = ap[sorted_indices]

    # Combine along columns: original voters + single pref + ranks + approval
    combined = torch.cat([voters_sorted, sp_sorted, rk_sorted.float(), ap_sorted], dim=1)

    # 5) Plot
    plt.figure(figsize=(12, 6))
    plt.imshow(combined, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title("Heatmap: [Voters sorted by argmax | SinglePref | Ranked | Approval]")
    plt.xlabel("Features (10 original, 10 single pref, 10 rank, 10 approval)")
    plt.ylabel("Voters (sorted by orientation they most prefer)")
    plt.show()


if __name__ == "__main__":
    demo()
Notes
torch.argmax(voters, dim=1) gets the index of the highest score for each voter.
torch.argsort(max_indices) sorts the voter indices by their preferred orientation index (i.e., grouping those who prefer orientation 0 first, then orientation 1, etc.).
We then reorder (voters, sp, rk, ap) using that same sorted_indices.
This way, the rows in the final heatmap are grouped by their most-preferred orientation.






You said:
This is looking great! Now let's modify the voter model so that each orientation will now be assigned a point in some number of dimensions (let's say 3 as a default) and let's spread out our orientations in a line in this 3-dimensional space. Now let's adjust the score preferences so that each voter favors orientations that are closer to their main preference with an inverse square relationship whose coefficient is randomly decided with a parameter for mean and standard deviation on the relative distance. The default mean should be half the distance from the left-most to right-most orientation points and the default standard deviation should be half that again. 
ChatGPT

one heatmap
Below is an updated VoterModel that assigns each orientation a position in a (default) 3-dimensional space arranged in a straight line. Then, when generating voters, we do the following:

Maintain the fraction of voters for each orientation as their “main preference.”
For each voter who “mainly prefers” orientation 
𝑖
i, draw a random coefficient (from a Normal distribution) to define how steeply the voter’s utility drops off with distance from orientation 
𝑖
i.
Compute scores for every orientation 
𝑗
j using an inverse-square-style function of the Euclidean distance from 
𝑖
i to 
𝑗
j.
Normalize each voter’s score vector to sum to 1.
In this setup:

We place the orientations in a straight line in 3D from 
(
0
,
0
,
0
)
(0,0,0) to 
(
𝑁
−
1
,
0
,
0
)
(N−1,0,0).
The total distance between left-most (orientation 0) and right-most (orientation 
𝑁
−
1
N−1) is 
(N - 1)
(N - 1).
By default, the mean of the random coefficient is set to half that distance, i.e., 
(
𝑁
−
1
)
/
2
(N−1)/2.
By default, the standard deviation is set to half that again, i.e., 
(
𝑁
−
1
)
/
4
(N−1)/4.
You can change the number of dimensions or the coefficient distribution parameters as needed.
python
Copy
Edit
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import matplotlib.pyplot as plt

class VoterModel(pl.LightningModule):
    """
    A PyTorch Lightning module that:
     - Places orientations in a D-dimensional line (default D=3).
     - Generates voters such that each orientation has a target fraction of main-preference voters.
     - Voters' scores follow an inverse-square relationship relative to their main preference.
     - Provides voting methods (single-preference, ranked, approval).
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _init_orientation_positions(num_orientations: int,
                                    num_dimensions: int = 3) -> torch.Tensor:
        """
        Place 'num_orientations' points in a straight line in 'num_dimensions' space.

        By default:
          - The left-most orientation is at (0, 0, 0, ...)
          - The right-most orientation is at (num_orientations - 1, 0, 0, ...)
          - Others are equally spaced along the x-axis only.

        Returns: [num_orientations, num_dimensions] float tensor of positions.
        """
        # We'll space them along the x-axis from 0..(N-1)
        coords = torch.zeros(num_orientations, num_dimensions)
        for i in range(num_orientations):
            coords[i, 0] = float(i)  # x = i, y = 0, z = 0, ...
        return coords

    def generate_voters(self,
                        orientations: dict,
                        num_voters: int,
                        num_dimensions: int = 3,
                        mean_coefficient: float = None,
                        std_coefficient: float = None) -> torch.Tensor:
        """
        Generate a tensor of voter preferences based on an inverse-square relationship
        in a D-dimensional space.

        Steps:
          1. We place each orientation i in D-dimensional space (line from (0,0,...,0) to (N-1, 0, ..., 0)).
          2. For each orientation i, we compute how many voters 'prefer' i using orientation fractions.
          3. For each such voter:
             a) Sample a random coefficient c ~ Normal(mean_coefficient, std_coefficient).
                (clamp it to a small positive min to avoid negative or zero).
             b) For each orientation j, compute the distance dist(i, j) in D-dim space.
             c) Score_j = c / (1 + dist^2).
             d) Normalize the scores to sum to 1.
          4. Concatenate all voters, then shuffle.

        Args:
            orientations (dict): A dictionary {orientation_name: fraction} that sums ~1.0
            num_voters (int): Number of total voters.
            num_dimensions (int): Dimensional space to place the orientations (default=3).
            mean_coefficient (float): Mean of the normal distribution for the random coefficient.
                                      If None, defaults to half the total line distance (i.e. (N-1)/2).
            std_coefficient  (float): Std of that distribution.
                                      If None, defaults to half of mean_coefficient => (N-1)/4.

        Returns:
            voters (torch.Tensor): [num_voters, num_orientations] where each row sums to 1.
        """
        # Extract orientation fractions
        orientation_names = list(orientations.keys())
        orientation_fractions = list(orientations.values())
        num_orientations = len(orientation_names)

        # 1) Generate orientation positions in D-dimensional line
        orientation_positions = self._init_orientation_positions(num_orientations, num_dimensions)
        # Distance between left-most and right-most
        total_line_dist = float(num_orientations - 1)  # along x-axis

        # Default mean & std
        if mean_coefficient is None:
            mean_coefficient = total_line_dist / 2.0
        if std_coefficient is None:
            std_coefficient = mean_coefficient / 2.0

        # 2) Convert fractions to integer counts
        counts = [round(frac * num_voters) for frac in orientation_fractions]
        total_assigned = sum(counts)
        if total_assigned < num_voters:
            idx = torch.argmax(torch.tensor(orientation_fractions)).item()
            counts[idx] += (num_voters - total_assigned)
        elif total_assigned > num_voters:
            idx = torch.argmax(torch.tensor(orientation_fractions)).item()
            counts[idx] -= (total_assigned - num_voters)

        # 3) Construct each voter's score vector
        all_voters = []
        for i, c in enumerate(counts):
            if c == 0:
                continue

            # For each of these c voters, main preference = orientation i.
            # We'll sample a random coefficient for each voter.
            # shape = [c]
            coeffs = torch.randn(c) * std_coefficient + mean_coefficient
            # clamp to ensure positivity
            coeffs = torch.clamp(coeffs, min=1e-6)

            # We'll compute distances from orientation i to *all* j once
            # orientation_positions[i] => shape [num_dimensions]
            # orientation_positions => shape [num_orientations, num_dimensions]
            diff = orientation_positions - orientation_positions[i]  # [num_orientations, D]
            dists_sq = (diff ** 2).sum(dim=1)  # [num_orientations], each is dist^2 from i->j

            # For each voter, we do Score_j = coeff / (1 + dist^2)
            # We'll broadcast over j
            # We'll create a base shape [num_orientations] then replicate for c voters
            # Then multiply by the c different coefficients
            # shape [num_orientations] => shape [c, num_orientations]
            base_scores = 1.0 / (1.0 + dists_sq)  # shape [num_orientations]
            # Expand to [c, num_orientations]
            base_scores_expanded = base_scores.unsqueeze(0).repeat(c, 1)
            # Multiply each row by that row's coefficient
            # shape [c]
            base_scores_expanded = base_scores_expanded * coeffs.unsqueeze(-1)

            # Now normalize each row to sum to 1
            row_sums = base_scores_expanded.sum(dim=1, keepdim=True)
            row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
            voter_scores = base_scores_expanded / row_sums  # shape [c, num_orientations]

            all_voters.append(voter_scores)

        voters = torch.cat(all_voters, dim=0)  # [num_voters, num_orientations]
        # Finally, shuffle to avoid grouping
        perm = torch.randperm(num_voters)
        voters = voters[perm]

        return voters
