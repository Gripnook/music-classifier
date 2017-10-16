package classifier.gaussian;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import classifier.SongClassifier;
import main.Genre;
import main.Song;
import numeric.Matrix;
import numeric.Stats;

/**
 * A classifier that classifies songs by computing the Gaussian distributions of
 * each genre and finding the one that best describes the song to be classified.
 * 
 * This classifier uses the feature vectors of each song independently to
 * compute the average and covariance of each genre. It then classifies each new
 * song by finding the genre where all its feature vectors will fit best.
 * 
 * @author Andrei Purcarus
 *
 */
public class TotalGaussianClassifier implements SongClassifier {
	private Map<Genre, Stats> stats = new HashMap<>();
	private Map<Genre, double[]> averages = new HashMap<>();
	private Map<Genre, double[][]> inverseCovs = new HashMap<>();

	public TotalGaussianClassifier() {
		for (Genre genre : Genre.class.getEnumConstants()) {
			stats.put(genre, new Stats(Song.FEATURES));
		}
	}

	@Override
	public void add(List<double[]> song, Genre genre) {
		for (double[] feature : song) {
			stats.get(genre).add(feature);
		}
	}

	@Override
	public void train() {
		for (Genre genre : Genre.class.getEnumConstants()) {
			averages.put(genre, stats.get(genre).average());
			inverseCovs.put(genre, Matrix.invert(stats.get(genre).covariance()));
		}
	}

	@Override
	public Genre classify(List<double[]> song) {
		// Minimizes the UNLL over all possible genres.
		Genre result = null;
		double unll = Double.MAX_VALUE;
		for (Genre genre : Genre.class.getEnumConstants()) {
			double[] average = averages.get(genre);
			double[][] inverseCov = inverseCovs.get(genre);
			double unllCandidate = 0;
			for (double[] feature : song) {
				for (int i = 0; i < Song.FEATURES; ++i) {
					for (int j = 0; j < Song.FEATURES; ++j) {
						unllCandidate += (feature[i] - average[i]) * inverseCov[i][j] * (feature[j] - average[j]);
					}
				}
			}
			if (unllCandidate < unll) {
				unll = unllCandidate;
				result = genre;
			}
		}
		return result;
	}

	@Override
	public void clear() {
		stats = new HashMap<>();
		averages = new HashMap<>();
		inverseCovs = new HashMap<>();
		for (Genre genre : Genre.class.getEnumConstants()) {
			stats.put(genre, new Stats(Song.FEATURES));
		}
	}
}
