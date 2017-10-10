package classifier;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import main.Genre;
import main.Song;

public class LearningClassifier implements SongClassifier {
	private SongClassifier classifier = new KNNClassifier(1);
	private double[] weights = new double[Song.FEATURES];
	private double[] prevWeights = new double[Song.FEATURES];
	private double success = 0.0;
	private Random rng = new Random();

	public LearningClassifier() {
		for (int i = 0; i < Song.FEATURES; ++i) {
			weights[i] = 1;
			prevWeights[i] = 1;
		}
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
			System.out.println("failed to learn");
			weights = prevWeights;
		}
		prevWeights = Arrays.copyOf(weights, Song.FEATURES);
		modify();
	}

	private void modify() {
		int index = rng.nextInt(Song.FEATURES);
		double factor = Math.pow(2, rng.nextDouble() - 0.5);
		weights[index] *= factor;
	}

	@Override
	public void add(List<double[]> song, Genre genre) {
		List<double[]> modifiedSong = new ArrayList<>();
		for (double[] feature : song) {
			double[] modifiedFeature = Arrays.copyOf(feature, Song.FEATURES);
			for (int i = 0; i < Song.FEATURES; ++i) {
				modifiedFeature[i] *= weights[i];
				modifiedSong.add(modifiedFeature);
			}
		}
		classifier.add(modifiedSong, genre);
	}

	@Override
	public void train() {
		classifier.train();
	}

	@Override
	public Genre classify(List<double[]> song) {
		List<double[]> modifiedSong = new ArrayList<>();
		for (double[] feature : song) {
			double[] modifiedFeature = Arrays.copyOf(feature, Song.FEATURES);
			for (int i = 0; i < Song.FEATURES; ++i) {
				modifiedFeature[i] *= weights[i];
				modifiedSong.add(modifiedFeature);
			}
		}
		return classifier.classify(modifiedSong);
	}

	@Override
	public void clear() {
		classifier.clear();
	}
}
