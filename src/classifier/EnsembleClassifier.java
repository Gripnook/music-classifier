package classifier;

import java.util.ArrayList;
import java.util.List;

import main.Genre;
import numeric.Plurality;

public class EnsembleClassifier implements SongClassifier {
	private List<SongClassifier> classifiers = new ArrayList<>();

	public EnsembleClassifier(List<SongClassifier> classifiers) {
		this.classifiers = classifiers;
	}

	@Override
	public void add(List<double[]> song, Genre genre) {
		for (SongClassifier classifier : classifiers) {
			classifier.add(song, genre);
		}
	}

	@Override
	public void train() {
		for (SongClassifier classifier : classifiers) {
			classifier.train();
		}
	}

	@Override
	public Genre classify(List<double[]> song) {
		Plurality<Genre> plurality = new Plurality<>();
		for (SongClassifier classifier : classifiers) {
			plurality.add(classifier.classify(song));
		}
		return plurality.vote();
	}

	@Override
	public void clear() {
		for (SongClassifier classifier : classifiers) {
			classifier.clear();
		}
	}
}
