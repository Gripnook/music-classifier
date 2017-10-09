public interface SongClassifier {
	public void add(double[] song, Genre genre);

	public void train();

	public Genre classify(double[] song);
	
	public void clear();
}
