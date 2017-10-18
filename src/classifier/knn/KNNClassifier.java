package classifier.knn;

import java.util.ArrayList;
import java.util.List;

import classifier.Entry;
import classifier.SongClassifier;
import main.Genre;
import main.Song;
import numeric.Plurality;

/**
 * A nearest neighbour classifier which takes the plurality vote of the k
 * nearest neighbours to classify new features. It then classifies songs by
 * taking the plurality vote of the individual feature classifications.
 * 
 * @author Andrei Purcarus
 *
 */
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
		// Uses a KD tree to speed up classification.
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
		Entry[] nearest = tree.nearest(k, feature, 100 * k);
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
