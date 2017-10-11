package classifier;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import main.Genre;
import numeric.Plurality;

public class DecisionForestClassifier implements SongClassifier {
	private int numTrees;
	private List<Entry> entries = new ArrayList<>();
	private List<DecisionTree> trees = null;
	private Random rng = new Random();

	public DecisionForestClassifier(int numTrees) {
		this.numTrees = numTrees;
	}

	@Override
	public void add(List<double[]> song, Genre genre) {
		for (double[] feature : song) {
			entries.add(new Entry(feature, genre));
		}
	}

	@Override
	public void train() {
		trees = new ArrayList<>();
		int subsetSize = 1 << 16;
		for (int i = 0; i < numTrees; ++i) {
			// Sample a random subset of the data without replacement.
			List<Entry> data = new ArrayList<>();
			for (int j = 0; j < subsetSize; ++j) {
				int index = rng.nextInt(entries.size());
				data.add(entries.get(index));
			}
			trees.add(new DecisionTree(data));
		}
	}

	@Override
	public Genre classify(List<double[]> song) {
		Plurality<Genre> plurality = new Plurality<>();
		for (double[] feature : song) {
			plurality.add(classify(feature));
		}
		return plurality.vote();
	}

	private Genre classify(double[] feature) {
		Plurality<Genre> plurality = new Plurality<>();
		for (DecisionTree tree : trees) {
			plurality.add(tree.classify(feature));
		}
		return plurality.vote();
	}

	@Override
	public void clear() {
		entries = new ArrayList<>();
		trees = null;
	}
}
