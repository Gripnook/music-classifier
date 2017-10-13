package classifier.knn;

import java.util.ArrayList;
import java.util.List;

import classifier.Entry;
import classifier.SongClassifier;
import main.Genre;
import main.Song;
import numeric.Plurality;

public class KNNClassifier implements SongClassifier {
	private int k;
	private List<Entry> songs = new ArrayList<>();
	private BestBinFirstKDTree tree = null;

	public KNNClassifier(int k) {
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
		Entry[] nearest = tree.nearest(k, feature);
		Plurality<Genre> plurality = new Plurality<>();
		for (Entry entry : nearest) {
			plurality.add(entry.genre);
		}
		return plurality.vote();
	}

	@Override
	public void clear() {
		songs = new ArrayList<>();
		tree = null;
	}
}
