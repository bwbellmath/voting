import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vote_random import VoterModel

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class VotingSimulator:
    """
    Class to simulate elections using various voting methods and generate visualizations.
    """

    def __init__(self, num_candidates=5, num_voters=1000):
        self.num_candidates = num_candidates
        self.num_voters = num_voters
        self.voter_model = VoterModel()
        self.candidate_names = [f"Candidate {chr(65+i)}" for i in range(num_candidates)]
        self.voters = None
        self.results = {}

    def generate_voters(self, distribution='uniform', variance=0.3):
        """
        Generate simulated voters with different preference distributions.

        Args:
            distribution: 'uniform', 'polarized', 'moderate', or 'skewed'
            variance: How much voters deviate from their orientation (0.0 to 1.0)
        """
        if distribution == 'uniform':
            # Equal support for all candidates
            orientations = {name: 1.0/self.num_candidates for name in self.candidate_names}
        elif distribution == 'polarized':
            # Two main candidates, rest split remaining
            main_fraction = 0.35
            other_fraction = (1.0 - 2*main_fraction) / max(1, self.num_candidates - 2)
            orientations = {}
            for i, name in enumerate(self.candidate_names):
                if i < 2:
                    orientations[name] = main_fraction
                else:
                    orientations[name] = other_fraction
        elif distribution == 'moderate':
            # Bell curve centered on middle candidate
            fractions = np.exp(-((np.arange(self.num_candidates) - self.num_candidates/2)**2) / 2)
            fractions = fractions / fractions.sum()
            orientations = {name: frac for name, frac in zip(self.candidate_names, fractions)}
        elif distribution == 'skewed':
            # Power law distribution
            fractions = 1.0 / (np.arange(self.num_candidates) + 1)
            fractions = fractions / fractions.sum()
            orientations = {name: frac for name, frac in zip(self.candidate_names, fractions)}
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        self.voters = self.voter_model.generate_voters(orientations, self.num_voters, variance)
        return self.voters

    def plurality_vote(self):
        """Standard plurality voting - each voter picks one candidate."""
        votes = self.voter_model.vote_single_preference(self.voters)
        totals = votes.sum(dim=0)
        winner_idx = torch.argmax(totals).item()

        self.results['plurality'] = {
            'totals': totals.numpy(),
            'winner': self.candidate_names[winner_idx],
            'winner_votes': totals[winner_idx].item(),
            'winner_percentage': (totals[winner_idx] / self.num_voters * 100).item()
        }
        return self.results['plurality']

    def approval_vote(self):
        """Approval voting - voters can approve multiple candidates."""
        approvals = self.voter_model.vote_approval(self.voters)
        totals = approvals.sum(dim=0)
        winner_idx = torch.argmax(totals).item()

        self.results['approval'] = {
            'totals': totals.numpy(),
            'winner': self.candidate_names[winner_idx],
            'winner_votes': totals[winner_idx].item(),
            'winner_percentage': (totals[winner_idx] / self.num_voters * 100).item()
        }
        return self.results['approval']

    def borda_count(self):
        """Borda count - points based on ranking position."""
        ranks = self.voter_model.vote_ranked(self.voters)
        # Convert ranks to points: 1st place gets num_candidates points, last gets 1
        points = self.num_candidates - ranks + 1
        totals = points.sum(dim=0)
        winner_idx = torch.argmax(totals).item()

        self.results['borda'] = {
            'totals': totals.numpy(),
            'winner': self.candidate_names[winner_idx],
            'winner_points': totals[winner_idx].item(),
            'average_points': (totals / self.num_voters).numpy()
        }
        return self.results['borda']

    def instant_runoff(self):
        """
        Instant runoff voting (ranked choice voting).
        Eliminate lowest-ranked candidates iteratively.
        """
        ranks = self.voter_model.vote_ranked(self.voters).clone()
        remaining_candidates = set(range(self.num_candidates))
        elimination_order = []

        while len(remaining_candidates) > 1:
            # Count first-choice votes among remaining candidates
            first_choices = torch.zeros(self.num_candidates)
            for voter_ranks in ranks:
                # Find the highest-ranked remaining candidate
                min_rank = float('inf')
                choice = -1
                for cand_idx in remaining_candidates:
                    if voter_ranks[cand_idx] < min_rank:
                        min_rank = voter_ranks[cand_idx]
                        choice = cand_idx
                if choice >= 0:
                    first_choices[choice] += 1

            # Check if anyone has majority
            max_votes = first_choices.max().item()
            if max_votes > self.num_voters / 2:
                winner_idx = torch.argmax(first_choices).item()
                break

            # Eliminate candidate with fewest first-choice votes
            min_votes = float('inf')
            eliminated = -1
            for cand_idx in remaining_candidates:
                if first_choices[cand_idx] < min_votes:
                    min_votes = first_choices[cand_idx]
                    eliminated = cand_idx

            remaining_candidates.remove(eliminated)
            elimination_order.append(self.candidate_names[eliminated])
        else:
            winner_idx = list(remaining_candidates)[0]

        self.results['instant_runoff'] = {
            'winner': self.candidate_names[winner_idx],
            'elimination_order': elimination_order,
            'rounds': len(elimination_order)
        }
        return self.results['instant_runoff']

    def condorcet_method(self):
        """
        Condorcet method - find candidate who wins all pairwise matchups.
        """
        # Create pairwise comparison matrix
        pairwise_wins = torch.zeros((self.num_candidates, self.num_candidates))

        for i in range(self.num_candidates):
            for j in range(i+1, self.num_candidates):
                # Count how many voters prefer i over j
                i_over_j = (self.voters[:, i] > self.voters[:, j]).sum()
                j_over_i = (self.voters[:, j] > self.voters[:, i]).sum()

                if i_over_j > j_over_i:
                    pairwise_wins[i, j] = 1
                elif j_over_i > i_over_j:
                    pairwise_wins[j, i] = 1

        # Find Condorcet winner (beats all others)
        wins_per_candidate = pairwise_wins.sum(dim=1)
        condorcet_winner = None
        if (wins_per_candidate == self.num_candidates - 1).any():
            winner_idx = torch.argmax(wins_per_candidate).item()
            condorcet_winner = self.candidate_names[winner_idx]

        self.results['condorcet'] = {
            'winner': condorcet_winner,
            'pairwise_wins': pairwise_wins.numpy(),
            'total_wins': wins_per_candidate.numpy()
        }
        return self.results['condorcet']

    def run_all_methods(self):
        """Run all voting methods and collect results."""
        print("Running plurality voting...")
        self.plurality_vote()

        print("Running approval voting...")
        self.approval_vote()

        print("Running Borda count...")
        self.borda_count()

        print("Running instant runoff voting...")
        self.instant_runoff()

        print("Running Condorcet method...")
        self.condorcet_method()

        return self.results

    def visualize_voter_preferences(self, save_path='img/voter_preferences.png'):
        """Create a heatmap of voter preferences."""
        # Sort voters by their top preference
        max_indices = torch.argmax(self.voters, dim=1)
        sorted_indices = torch.argsort(max_indices)
        voters_sorted = self.voters[sorted_indices].numpy()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(voters_sorted, cmap='YlOrRd', cbar_kws={'label': 'Preference Score'},
                    xticklabels=self.candidate_names, yticklabels=False)
        ax.set_xlabel('Candidates', fontsize=12)
        ax.set_ylabel(f'Voters (n={self.num_voters}, sorted by top preference)', fontsize=12)
        ax.set_title('Voter Preference Heatmap', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved voter preferences visualization to {save_path}")
        plt.close()

    def visualize_rankings_distribution(self, save_path='img/rankings_distribution.png'):
        """Visualize the distribution of rankings for each candidate."""
        ranks = self.voter_model.vote_ranked(self.voters).numpy()

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, name in enumerate(self.candidate_names):
            ax = axes[i]
            rank_counts = np.bincount(ranks[:, i].astype(int), minlength=self.num_candidates+1)[1:]

            colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, self.num_candidates))
            ax.bar(range(1, self.num_candidates+1), rank_counts, color=colors, edgecolor='black')
            ax.set_xlabel('Rank Position', fontsize=10)
            ax.set_ylabel('Number of Voters', fontsize=10)
            ax.set_title(f'{name}', fontsize=12, fontweight='bold')
            ax.set_xticks(range(1, self.num_candidates+1))
            ax.grid(axis='y', alpha=0.3)

        # Remove extra subplot if odd number of candidates
        if len(self.candidate_names) < len(axes):
            axes[-1].remove()

        fig.suptitle('Distribution of Rankings for Each Candidate',
                     fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved rankings distribution to {save_path}")
        plt.close()

    def visualize_method_comparison(self, save_path='img/method_comparison.png'):
        """Create a comparison of results across all voting methods."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plurality
        ax = axes[0, 0]
        plurality_data = self.results['plurality']['totals']
        colors_plurality = ['gold' if self.candidate_names[i] == self.results['plurality']['winner']
                           else 'skyblue' for i in range(self.num_candidates)]
        ax.bar(self.candidate_names, plurality_data, color=colors_plurality, edgecolor='black')
        ax.set_ylabel('Number of Votes', fontsize=10)
        ax.set_title('Plurality Voting', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        winner_text = f"Winner: {self.results['plurality']['winner']}\n({self.results['plurality']['winner_percentage']:.1f}%)"
        ax.text(0.02, 0.98, winner_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Approval
        ax = axes[0, 1]
        approval_data = self.results['approval']['totals']
        colors_approval = ['gold' if self.candidate_names[i] == self.results['approval']['winner']
                          else 'lightcoral' for i in range(self.num_candidates)]
        ax.bar(self.candidate_names, approval_data, color=colors_approval, edgecolor='black')
        ax.set_ylabel('Number of Approvals', fontsize=10)
        ax.set_title('Approval Voting', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        winner_text = f"Winner: {self.results['approval']['winner']}\n({self.results['approval']['winner_percentage']:.1f}%)"
        ax.text(0.02, 0.98, winner_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Borda Count
        ax = axes[1, 0]
        borda_data = self.results['borda']['totals']
        colors_borda = ['gold' if self.candidate_names[i] == self.results['borda']['winner']
                       else 'lightgreen' for i in range(self.num_candidates)]
        ax.bar(self.candidate_names, borda_data, color=colors_borda, edgecolor='black')
        ax.set_ylabel('Total Points', fontsize=10)
        ax.set_title('Borda Count', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        avg_points = self.results['borda']['average_points']
        winner_idx = np.argmax(avg_points)
        winner_text = f"Winner: {self.results['borda']['winner']}\n(Avg: {avg_points[winner_idx]:.2f} pts)"
        ax.text(0.02, 0.98, winner_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Instant Runoff
        ax = axes[1, 1]
        # Show elimination order
        elimination_data = list(reversed(self.results['instant_runoff']['elimination_order']))
        elimination_data.insert(0, self.results['instant_runoff']['winner'])
        positions = range(len(elimination_data))
        colors_irv = ['gold'] + ['salmon'] * len(self.results['instant_runoff']['elimination_order'])
        ax.barh(positions, [len(positions) - i for i in positions], color=colors_irv, edgecolor='black')
        ax.set_yticks(positions)
        ax.set_yticklabels(elimination_data)
        ax.set_xlabel('Survival Order', fontsize=10)
        ax.set_title('Instant Runoff Voting (RCV)', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        winner_text = f"Winner: {self.results['instant_runoff']['winner']}\n({self.results['instant_runoff']['rounds']} rounds)"
        ax.text(0.98, 0.98, winner_text, transform=ax.transAxes, ha='right',
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        fig.suptitle('Comparison of Voting Methods', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved method comparison to {save_path}")
        plt.close()

    def visualize_condorcet_matrix(self, save_path='img/condorcet_matrix.png'):
        """Visualize pairwise comparison matrix for Condorcet method."""
        pairwise_wins = self.results['condorcet']['pairwise_wins']

        fig, ax = plt.subplots(figsize=(10, 8))

        # Create a symmetric matrix showing win percentages
        win_matrix = np.zeros((self.num_candidates, self.num_candidates))
        for i in range(self.num_candidates):
            for j in range(self.num_candidates):
                if i != j:
                    i_over_j = (self.voters[:, i] > self.voters[:, j]).sum().item()
                    win_matrix[i, j] = i_over_j / self.num_voters * 100

        sns.heatmap(win_matrix, annot=True, fmt='.1f', cmap='RdYlGn', center=50,
                    xticklabels=self.candidate_names, yticklabels=self.candidate_names,
                    cbar_kws={'label': '% of voters preferring row over column'}, ax=ax)

        ax.set_xlabel('Opposed Candidate', fontsize=12)
        ax.set_ylabel('Preferred Candidate', fontsize=12)

        condorcet_winner = self.results['condorcet']['winner']
        title = 'Condorcet Pairwise Comparison Matrix'
        if condorcet_winner:
            title += f'\nCondorcet Winner: {condorcet_winner}'
        else:
            title += '\nNo Condorcet Winner (Voting Paradox)'
        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Condorcet matrix to {save_path}")
        plt.close()

    def create_results_summary(self, save_path='img/results_summary.txt'):
        """Create a text summary of all results."""
        with open(save_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("VOTING SIMULATION RESULTS SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Number of Voters: {self.num_voters}\n")
            f.write(f"Number of Candidates: {self.num_candidates}\n")
            f.write(f"Candidates: {', '.join(self.candidate_names)}\n\n")

            f.write("-" * 80 + "\n")
            f.write("PLURALITY VOTING\n")
            f.write("-" * 80 + "\n")
            f.write(f"Winner: {self.results['plurality']['winner']}\n")
            f.write(f"Votes: {self.results['plurality']['winner_votes']:.0f} ({self.results['plurality']['winner_percentage']:.2f}%)\n")
            f.write("\nAll candidates:\n")
            for i, name in enumerate(self.candidate_names):
                votes = self.results['plurality']['totals'][i]
                pct = votes / self.num_voters * 100
                f.write(f"  {name}: {votes:.0f} votes ({pct:.2f}%)\n")

            f.write("\n" + "-" * 80 + "\n")
            f.write("APPROVAL VOTING\n")
            f.write("-" * 80 + "\n")
            f.write(f"Winner: {self.results['approval']['winner']}\n")
            f.write(f"Approvals: {self.results['approval']['winner_votes']:.0f} ({self.results['approval']['winner_percentage']:.2f}%)\n")
            f.write("\nAll candidates:\n")
            for i, name in enumerate(self.candidate_names):
                approvals = self.results['approval']['totals'][i]
                pct = approvals / self.num_voters * 100
                f.write(f"  {name}: {approvals:.0f} approvals ({pct:.2f}%)\n")

            f.write("\n" + "-" * 80 + "\n")
            f.write("BORDA COUNT\n")
            f.write("-" * 80 + "\n")
            f.write(f"Winner: {self.results['borda']['winner']}\n")
            f.write(f"Points: {self.results['borda']['winner_points']:.0f}\n")
            f.write("\nAll candidates:\n")
            for i, name in enumerate(self.candidate_names):
                points = self.results['borda']['totals'][i]
                avg = self.results['borda']['average_points'][i]
                f.write(f"  {name}: {points:.0f} points (avg: {avg:.2f})\n")

            f.write("\n" + "-" * 80 + "\n")
            f.write("INSTANT RUNOFF VOTING (RCV)\n")
            f.write("-" * 80 + "\n")
            f.write(f"Winner: {self.results['instant_runoff']['winner']}\n")
            f.write(f"Rounds: {self.results['instant_runoff']['rounds']}\n")
            f.write("\nElimination order:\n")
            for i, candidate in enumerate(self.results['instant_runoff']['elimination_order'], 1):
                f.write(f"  Round {i}: {candidate} eliminated\n")

            f.write("\n" + "-" * 80 + "\n")
            f.write("CONDORCET METHOD\n")
            f.write("-" * 80 + "\n")
            if self.results['condorcet']['winner']:
                f.write(f"Condorcet Winner: {self.results['condorcet']['winner']}\n")
                f.write("(Beats all other candidates in pairwise matchups)\n")
            else:
                f.write("No Condorcet Winner (Voting Paradox)\n")

            f.write("\nPairwise matchup wins:\n")
            for i, name in enumerate(self.candidate_names):
                wins = int(self.results['condorcet']['total_wins'][i])
                f.write(f"  {name}: {wins} wins\n")

            f.write("\n" + "=" * 80 + "\n")

        print(f"Saved results summary to {save_path}")


def main():
    """Main function to run the simulation and generate all visualizations."""

    print("\n" + "="*80)
    print("VOTING SIMULATION AND VISUALIZATION")
    print("="*80 + "\n")

    # Create simulator with 5 candidates and 1000 voters
    sim = VotingSimulator(num_candidates=5, num_voters=1000)

    # Test different voter distributions
    distributions = ['uniform', 'polarized', 'moderate', 'skewed']

    for dist in distributions:
        print(f"\n{'='*80}")
        print(f"SIMULATION: {dist.upper()} DISTRIBUTION")
        print(f"{'='*80}\n")

        # Generate voters
        print(f"Generating {sim.num_voters} voters with {dist} distribution...")
        sim.generate_voters(distribution=dist, variance=0.3)

        # Run all voting methods
        sim.run_all_methods()

        # Create visualizations
        print("\nGenerating visualizations...")
        sim.visualize_voter_preferences(f'img/voter_preferences_{dist}.png')
        sim.visualize_rankings_distribution(f'img/rankings_distribution_{dist}.png')
        sim.visualize_method_comparison(f'img/method_comparison_{dist}.png')
        sim.visualize_condorcet_matrix(f'img/condorcet_matrix_{dist}.png')
        sim.create_results_summary(f'img/results_summary_{dist}.txt')

        print(f"\n{'-'*80}")
        print(f"RESULTS SUMMARY for {dist.upper()} distribution:")
        print(f"{'-'*80}")
        print(f"Plurality winner:      {sim.results['plurality']['winner']}")
        print(f"Approval winner:       {sim.results['approval']['winner']}")
        print(f"Borda count winner:    {sim.results['borda']['winner']}")
        print(f"Instant runoff winner: {sim.results['instant_runoff']['winner']}")
        print(f"Condorcet winner:      {sim.results['condorcet']['winner'] or 'None (paradox)'}")
        print(f"{'-'*80}\n")

    print("\n" + "="*80)
    print("SIMULATION COMPLETE!")
    print("="*80)
    print("\nAll visualizations and summaries have been saved to the 'img/' directory.")
    print("Check the following files:")
    print("  - voter_preferences_*.png: Heatmaps of voter preferences")
    print("  - rankings_distribution_*.png: Distribution of rankings for each candidate")
    print("  - method_comparison_*.png: Comparison of results across voting methods")
    print("  - condorcet_matrix_*.png: Pairwise comparison matrices")
    print("  - results_summary_*.txt: Text summaries of results")
    print("\n")


if __name__ == "__main__":
    main()
