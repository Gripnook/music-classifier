package classifier;

import main.Genre;

class Entry {
	public double[] feature;
	public Genre genre;

	public Entry(double[] feature, Genre genre) {
		this.feature = feature;
		this.genre = genre;
	}
}
