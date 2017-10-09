import java.util.HashMap;
import java.util.Map;

public class GaussianClassifier implements SongClassifier {
	private Map<Genre, Stats> stats = new HashMap<>();
	private Map<Genre, double[]> averages = new HashMap<>();
	private Map<Genre, double[][]> inverseCovs = new HashMap<>();

	public GaussianClassifier() {
		for (Genre genre : Genre.class.getEnumConstants()) {
			stats.put(genre, new Stats());
		}
	}

	@Override
	public void add(double[] song, Genre genre) {
		stats.get(genre).add(song);
	}

	@Override
	public void train() {
		for (Genre genre : Genre.class.getEnumConstants()) {
			averages.put(genre, stats.get(genre).average());
			inverseCovs.put(genre, Matrix.invert(stats.get(genre).covariance()));
		}
	}

	@Override
	public Genre classify(double[] song) {
		Genre result = null;
		double unll = Double.MAX_VALUE;
		for (Genre genre : Genre.class.getEnumConstants()) {
			double[] average = averages.get(genre);
			double[][] inverseCov = inverseCovs.get(genre);
			double unllCandidate = 0;
			for (int i = 0; i < Song.DATA_SIZE; ++i) {
				for (int j = 0; j < Song.DATA_SIZE; ++j) {
					unllCandidate += (song[i] - average[i]) * (song[j] - average[j]) * inverseCov[i][j];
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
			stats.put(genre, new Stats());
		}
	}
}
