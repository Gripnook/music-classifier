package classifier;

import java.util.List;

import main.Genre;

public interface SongClassifier {
	public void add(List<double[]> song, Genre genre);

	public void train();

	public Genre classify(List<double[]> song);

	public void clear();
}
