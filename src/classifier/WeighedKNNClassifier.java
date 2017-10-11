package classifier;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import main.Genre;
import main.Song;
import numeric.Mahalanobis;
import numeric.Stats;

public class WeighedKNNClassifier implements SongClassifier {
	private int k;
	private List<Entry> songs = new ArrayList<>();
	private BestBinFirstKDTree tree = null;
	private Mahalanobis mahalanobis = null;

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
		tree = new BestBinFirstKDTree(songs, Song.FEATURES);
		List<double[]> data = new ArrayList<>();
		for (Entry song : songs) {
			data.add(song.feature);
		}
		mahalanobis = new Mahalanobis(data, Song.FEATURES);
	}

	private static final Genre[] genres = Genre.class.getEnumConstants();
	private static final Map<Genre, Integer> genreIndices = new HashMap<>();
	static {
		for (int i = 0; i < genres.length; ++i) {
			genreIndices.put(genres[i], i);
		}
	}

	@Override
	public Genre classify(List<double[]> song) {
		Stats stats = new Stats(genres.length);
		for (double[] feature : song) {
			stats.add(classify(feature));
		}
		return maximumLikelihood(stats.average());
	}

	private double[] classify(double[] feature) {
		Entry[] nearest = tree.nearest(k, feature);
		double[] probabilities = new double[genres.length];
		for (Entry entry : nearest) {
			double dist = distance(entry.feature, feature);
			if (dist == 0) {
				double[] guaranteed = new double[genres.length];
				guaranteed[genreIndices.get(entry.genre)] = 1.0;
				return guaranteed;
			}
			probabilities[genreIndices.get(entry.genre)] += 1 / (dist * dist);
		}
		return normalize(probabilities);
	}

	private Genre maximumLikelihood(double[] probabilities) {
		double max = 0;
		Genre result = null;
		for (int i = 0; i < genres.length; ++i) {
			double probability = probabilities[i];
			if (probability > max) {
				max = probability;
				result = genres[i];
			}
		}
		return result;
	}

	private double[] normalize(double[] probabilities) {
		double sum = 0;
		for (double probability : probabilities) {
			sum += probability;
		}
		for (int i = 0; i < genres.length; ++i) {
			probabilities[i] /= sum;
		}
		return probabilities;
	}

	private double distance(double[] lhs, double[] rhs) {
		return mahalanobis.distance(lhs, rhs);
	}

	@Override
	public void clear() {
		songs = new ArrayList<>();
		tree = null;
		mahalanobis = null;
	}
}
