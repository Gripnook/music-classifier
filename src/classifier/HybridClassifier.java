package classifier;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import main.Genre;

public class HybridClassifier implements SongClassifier {
	private List<SongClassifier> classifiers = new ArrayList<>();

	public HybridClassifier(List<SongClassifier> classifiers) {
		this.classifiers = classifiers;
	}

	@Override
	public void add(List<double[]> song, Genre genre) {
		for (SongClassifier classifier : classifiers)
			classifier.add(song, genre);
	}

	@Override
	public void train() {
		for (SongClassifier classifier : classifiers)
			classifier.train();
	}

	@Override
	public Genre classify(List<double[]> song) {
		Map<Genre, Integer> counts = new HashMap<>();
		for (SongClassifier classifier : classifiers) {
			Genre genre = classifier.classify(song);
			Integer count = counts.get(genre);
			if (count == null) {
				count = new Integer(0);
			}
			++count;
			if (count > classifiers.size() / 2)
				return genre;
			counts.put(genre, count);
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
		for (SongClassifier classifier : classifiers)
			classifier.clear();
	}
}
