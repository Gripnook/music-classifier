package classifier.knn;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import classifier.Entry;
import classifier.SongClassifier;
import main.Genre;
import main.Song;
import numeric.Stats;

public class WeighedKNNClassifier implements SongClassifier {
	private int k;
	private List<Entry> songs = new ArrayList<>();
	private KDTree tree = null;

	public WeighedKNNClassifier(int k) {
		this.k = k;
	}

	@Override
	public void add(List<double[]> song, Genre genre) {
		for (double[] feature : song) {
			songs.add(new Entry(feature, genre));
		}
	}

	@Override
	public void train() {
		tree = new KDTree(songs, Song.FEATURES);
		List<double[]> data = new ArrayList<>();
		for (Entry song : songs) {
			data.add(song.feature);
		}
	}

	private static final Genre[] GENRES = Genre.class.getEnumConstants();
	private static final Map<Genre, Integer> GENRE_INDICES = new HashMap<>();
	static {
		for (int i = 0; i < GENRES.length; ++i) {
			GENRE_INDICES.put(GENRES[i], i);
		}
	}

	@Override
	public Genre classify(List<double[]> song) {
		Stats stats = new Stats(GENRES.length);
		for (double[] feature : song) {
			stats.add(classify(feature));
		}
		return maximumLikelihood(stats.average());
	}

	private double[] classify(double[] feature) {
		Entry[] nearest = tree.nearest(k, feature);
		double[] probabilities = new double[GENRES.length];
		for (Entry entry : nearest) {
			double dist = distance(entry.feature, feature);
			if (dist == 0) {
				double[] guaranteed = new double[GENRES.length];
				guaranteed[GENRE_INDICES.get(entry.genre)] = 1.0;
				return guaranteed;
			}
			probabilities[GENRE_INDICES.get(entry.genre)] += 1 / (dist * dist);
		}
		return normalize(probabilities);
	}

	private Genre maximumLikelihood(double[] probabilities) {
		double max = 0;
		Genre result = null;
		for (int i = 0; i < GENRES.length; ++i) {
			double probability = probabilities[i];
			if (probability > max) {
				max = probability;
				result = GENRES[i];
			}
		}
		return result;
	}

	private double[] normalize(double[] probabilities) {
		double sum = 0;
		for (double probability : probabilities) {
			sum += probability;
		}
		for (int i = 0; i < GENRES.length; ++i) {
			probabilities[i] /= sum;
		}
		return probabilities;
	}

	private double distance(double[] lhs, double[] rhs) {
		double result = 0;
		for (int i = 0; i < Song.FEATURES; ++i) {
			result += (lhs[i] - rhs[i]) * (lhs[i] - rhs[i]);
		}
		return Math.sqrt(result);
	}

	@Override
	public void clear() {
		songs = new ArrayList<>();
		tree = null;
	}
}
