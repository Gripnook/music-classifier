package classifier;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import main.Genre;
import main.Song;
import numeric.PCA;
import numeric.Stats;

public class PCAKNNClassifier implements SongClassifier {
	private int k, dataSize;
	private Stats stats = new Stats(Song.FEATURES);
	private List<Entry> entries = new ArrayList<>();
	private PCA pca = null;
	private KDTree tree = null;

	public PCAKNNClassifier(int k, int dataSize) {
		this.k = k;
		this.dataSize = dataSize;
	}

	@Override
	public void add(List<double[]> song, Genre genre) {
		for (double[] feature : song) {
			stats.add(feature);
			entries.add(new Entry(feature, genre));
		}
	}

	@Override
	public void train() {
		pca = new PCA(stats.covariance(), dataSize);
		for (Entry entry : entries) {
			entry.feature = pca.transform(entry.feature);
		}
		tree = new KDTree(entries, dataSize);
	}

	@Override
	public Genre classify(List<double[]> song) {
		Map<Genre, Integer> counts = new HashMap<>();
		for (double[] feature : song) {
			Genre genre = classify(pca.transform(feature));
			Integer count = counts.get(genre);
			if (count == null) {
				count = new Integer(0);
			}
			++count;
			counts.put(genre, count);
		}
		return consensus(counts);
	}

	private Genre classify(double[] feature) {
		Entry[] nearest = tree.nearest(k, feature);
		Map<Genre, Integer> counts = new HashMap<>();
		for (Entry entry : nearest) {
			Integer count = counts.get(entry.genre);
			if (count == null) {
				count = new Integer(0);
			}
			++count;
			counts.put(entry.genre, count);
		}
		return consensus(counts);
	}

	private Genre consensus(Map<Genre, Integer> counts) {
		int max = 0;
		Genre result = null;
		for (Genre genre : counts.keySet()) {
			int count = counts.get(genre);
			if (count > max) {
				max = count;
				result = genre;
			}
		}
		return result;
	}

	@Override
	public void clear() {
		stats = new Stats(Song.FEATURES);
		entries = new ArrayList<>();
		pca = null;
		tree = null;
	}
}
