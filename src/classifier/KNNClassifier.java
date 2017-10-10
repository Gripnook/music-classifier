package classifier;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import main.Genre;
import main.Song;

public class KNNClassifier implements SongClassifier {
	private int k;
	private List<Entry> songs = new ArrayList<>();
	private KDTree tree = null;

	public KNNClassifier(int k) {
		this.k = k;
	}

	@Override
	public void add(List<double[]> song, Genre genre) {
		for (double[] feature : song)
			songs.add(new Entry(feature, genre));
	}

	@Override
	public void train() {
		tree = new KDTree(songs, Song.FEATURES);
	}

	@Override
	public Genre classify(List<double[]> song) {
		Map<Genre, Integer> counts = new HashMap<>();
		for (double[] feature : song) {
			Genre genre = classify(feature);
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
		songs = new ArrayList<>();
		tree = null;
	}
}
