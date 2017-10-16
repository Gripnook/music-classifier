package numeric;

/**
 * A helper class which keeps track of statistical data such as the average and
 * covariance of a set of N-dimensional vectors.
 * 
 * @author Andrei Purcarus
 *
 */
public class Stats {
	private int dataSize;
	private int dataCount = 0;
	private double[] sum;
	private double[][] sumOfProducts;

	/**
	 * Creates a new statistics object for data of the given size.
	 * 
	 * @param dataSize
	 */
	public Stats(int dataSize) {
		this.dataSize = dataSize;
		sum = new double[dataSize];
		sumOfProducts = new double[dataSize][dataSize];
	}

	/**
	 * Adds the data to the set.
	 * 
	 * @param data
	 */
	public void add(double[] data) {
		++dataCount;
		for (int i = 0; i < dataSize; ++i) {
			sum[i] += data[i];
			for (int j = 0; j < dataSize; ++j) {
				sumOfProducts[i][j] += data[i] * data[j];
			}
		}
	}

	/**
	 * Gets the current average of the data. This requires at least one data
	 * point.
	 * 
	 * @return
	 */
	public double[] average() {
		if (dataCount == 0) {
			return null;
		}

		double[] result = new double[dataSize];
		for (int i = 0; i < dataSize; ++i) {
			result[i] = sum[i] / dataCount;
		}
		return result;
	}

	/**
	 * Gets the current covariance matrix of the data. This requires at least
	 * two data points.
	 * 
	 * @return
	 */
	public double[][] covariance() {
		if (dataCount == 0 || dataCount == 1) {
			return null;
		}

		double[] average = average();
		double[][] result = new double[dataSize][dataSize];
		for (int i = 0; i < dataSize; ++i) {
			for (int j = 0; j < dataSize; ++j) {
				result[i][j] = (sumOfProducts[i][j] - dataCount * average[i] * average[j]) / (dataCount - 1);
			}
		}
		return result;
	}
}
