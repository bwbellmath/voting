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
       - Approval votes
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
            # Add leftover to the orientation with the largest fraction
            idx = torch.argmax(torch.tensor(orientation_fractions)).item()
            counts[idx] += (num_voters - total_assigned)
        elif total_assigned > num_voters:
            # Remove excess from the orientation with the largest fraction
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
                # Avoid division by zero if entire row is 0
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

        Args:
            voters (torch.Tensor): [N, M] scores

        Returns:
            Torch Tensor of same shape [N, M], one-hot vectors.
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

        Args:
            voters (torch.Tensor): [N, M] scores

        Returns:
            Torch Tensor [N, M] with integer ranks in each row.
        """
        sorted_indices = torch.argsort(voters, dim=1, descending=True)
        # sorted_indices[i] = [idx_of_highest_score, idx_of_2nd_highest_score, ...]
        # ranks[i][j] = rank of orientation j (1-based)
        ranks = torch.argsort(sorted_indices, dim=1) + 1
        return ranks

    def vote_approval(self, voters: torch.Tensor) -> torch.Tensor:
        """
        NEW: Convert a [num_voters, num_orientations] score matrix into 
        approval votes, based on each voter's mean and standard deviation.

        For each voter, we mark an orientation with 1 if:
            score > (mean_score + std_score) 
        else 0.

        If std_score = 0 for a voter (i.e., all orientations have same score),
        then no orientation will exceed (mean + std), thus that voter approves 
        no candidates (all 0).

        Args:
            voters (torch.Tensor): [N, M] scores

        Returns:
            Torch Tensor [N, M] of 0/1 approvals.
        """
        # Compute each row's mean and std
        row_means = voters.mean(dim=1, keepdim=True)   # shape [N, 1]
        row_stds  = voters.std(dim=1, keepdim=True)    # shape [N, 1]

        threshold = row_means + row_stds
        approval = (voters > threshold).float()
        return approval


def lexicographic_sort(voters: torch.Tensor) -> torch.Tensor:
    """
    Sort the rows of `voters` lexicographically, based on the scores
    from first dimension to last dimension.

    Args:
        voters (torch.Tensor): [N, M] float scores.

    Returns:
        Torch Tensor [N, M], sorted lex order by row.
    """
    # Move to CPU if needed
    cpu_voters = voters.cpu()
    # Convert to list of lists
    voters_list = cpu_voters.tolist()
    # Sort lexicographically (Python default for lists is lex ordering)
    sorted_list = sorted(voters_list)
    # Convert back to torch
    return torch.tensor(sorted_list, dtype=voters.dtype)


def demo():
    """
    Demo function that:
    1. Creates a config of 10 orientations (each 0.1)
    2. Generates 100 voters with variance=0.1
    3. Derives single-preference, ranked, and approval votes
    4. Plots a heatmap of (sorted) original voters and these derived votes
    """
    # Instantiate model
    model = VoterModel()

    # Example orientation config for 10 orientations
    # Summation is 1.0
    orientations = {
        f"orientation_{i+1}": 0.1 for i in range(10)
    }

    num_voters = 100
    variance = 0.4

    # 1) Generate voters
    voters = model.generate_voters(orientations, num_voters, variance)

    # 2) Compute different voting methods
    sp = model.vote_single_preference(voters)  # single-preference
    rk = model.vote_ranked(voters)            # ranked
    ap = model.vote_approval(voters)          # approval

    # 3) Sort by argmax of each voter's row
    max_indices = torch.argmax(voters, dim=1)  # shape [num_voters]
    sorted_indices = torch.argsort(max_indices)  # ascending sort by orientation index
    # If you'd like to group voters who prefer orientation 0, then orientation 1, etc.
    # This will place all orientation-0-lovers first, then orientation-1, etc.
    # Ties do not matter here since argmax is a single index.

    # 4) Reorder all data in the same way
    voters_sorted = voters[sorted_indices]
    sp_sorted = sp[sorted_indices]
    rk_sorted = rk[sorted_indices]
    ap_sorted = ap[sorted_indices]

    # 4) Combine into a single large matrix for a single heatmap
    #    We'll place them side by side along columns.
    #    final shape = [N, M + M + M + M] = [N, 4*M], if M is 10 for the original, 
    #    but note ranks are also shape [N, M], so that's consistent.
    rk_sorted_frac = rk_sorted.float()/rk_sorted.shape[1]
    combined = torch.cat([voters_sorted, sp_sorted, rk_sorted_frac, ap_sorted], dim=1)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.imshow(combined, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title("Heatmap: [Sorted Voters | SinglePref | Ranked | Approval]")
    plt.xlabel("Features (10 for original, then 10 for single pref, then 10 for rank, then 10 for approval)")
    plt.ylabel("Voters (lexicographically sorted by original scores)")
    plt.show()


if __name__ == "__main__":
    demo()
