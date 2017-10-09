import java.util.HashMap;
import java.util.List;
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
	public void add(List<double[]> song, Genre genre) {
		Stats songStats = new Stats();
		for (double[] feature : song)
			songStats.add(feature);
		stats.get(genre).add(songStats.average());
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
		Stats songStats = new Stats();
		for (double[] feature : song)
			songStats.add(feature);
		double[] feature = songStats.average();

		Genre result = null;
		double unll = Double.MAX_VALUE;
		for (Genre genre : Genre.class.getEnumConstants()) {
			double[] average = averages.get(genre);
			double[][] inverseCov = inverseCovs.get(genre);
			double unllCandidate = 0;
			for (int i = 0; i < Song.DATA_SIZE; ++i) {
				for (int j = 0; j < Song.DATA_SIZE; ++j) {
					unllCandidate += (feature[i] - average[i]) * (feature[j] - average[j]) * inverseCov[i][j];
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
