import java.util.List;

public interface SongClassifier {
	public void add(List<double[]> song, Genre genre);

	public void train();

	public Genre classify(List<double[]> song);

	public void clear();
}
