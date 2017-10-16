package classifier;

import java.util.List;

import main.Genre;

/**
 * An interface for classifying songs into different genres.
 * 
 * @author Andrei Purcarus
 *
 */
public interface SongClassifier {
	/**
	 * Adds the current (song, genre) pair to the training set.
	 * 
	 * @param song
	 * @param genre
	 */
	public void add(List<double[]> song, Genre genre);

	/**
	 * Trains the classifier. This method must be called after the training data
	 * has been added and before new data can be classified.
	 */
	public void train();

	/**
	 * Returns the most likely genre for the given song. The classifier must be
	 * trained before this method is called.
	 * 
	 * @param song
	 * @return
	 */
	public Genre classify(List<double[]> song);

	/**
	 * Clears the training data.
	 */
	public void clear();
}
