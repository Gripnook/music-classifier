package classifier;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import main.Genre;
import main.Song;

public class LearningClassifier implements SongClassifier {
	private WeighedKNNClassifier classifier = new WeighedKNNClassifier(7);
	private double[] weights = new double[Song.FEATURES];
	private double[] prevWeights = new double[Song.FEATURES];
	private double success = 0.0;
	private Random rng = new Random();

	public LearningClassifier() {
		for (int i = 0; i < Song.FEATURES; ++i) {
			weights[i] = 1.0;
			prevWeights[i] = 1.0;
		}
		classifier.setWeights(weights);
	}

	public void learn(double success) {
		System.out.println("current weights: ");
		for (int i = 0; i < Song.FEATURES; ++i) {
			System.out.print(weights[i] + " ");
		}
		System.out.println();
		if (success > this.success) {
			System.out.println("learned from " + this.success + " to " + success);
			this.success = success;
		} else {
			System.out.println("failed to learn, still " + this.success);
			weights = prevWeights;
		}
		modify();
	}

	private void modify() {
		int index = rng.nextInt(Song.FEATURES);
		double factor = Math.pow(2, rng.nextDouble() - 0.5);
		prevWeights = Arrays.copyOf(weights, Song.FEATURES);
		weights[index] *= factor;
		classifier.setWeights(weights);
	}

	@Override
	public void add(List<double[]> song, Genre genre) {
		classifier.add(song, genre);
	}

	@Override
	public void train() {
		classifier.train();
	}

	@Override
	public Genre classify(List<double[]> song) {
		return classifier.classify(song);
	}

	@Override
	public void clear() {
		classifier.clear();
	}
}
