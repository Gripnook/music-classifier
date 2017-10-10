package numeric;

public class Stats {
	private int dataSize;
	private int dataCount = 0;
	private double[] sum;
	private double[][] sumOfProducts;

	public Stats(int dataSize) {
		this.dataSize = dataSize;
		sum = new double[dataSize];
		sumOfProducts = new double[dataSize][dataSize];
	}

	public void add(double[] data) {
		++dataCount;
		for (int i = 0; i < dataSize; ++i) {
			sum[i] += data[i];
			for (int j = 0; j < dataSize; ++j) {
				sumOfProducts[i][j] += data[i] * data[j];
			}
		}
	}

	public double[] average() {
		if (dataCount == 0)
			return null;

		double[] result = new double[dataSize];
		for (int i = 0; i < dataSize; ++i) {
			result[i] = sum[i] / dataCount;
		}
		return result;
	}

	public double[][] covariance() {
		if (dataCount == 0)
			return null;

		double[] average = average();
		double[][] result = new double[dataSize][dataSize];
		for (int i = 0; i < dataSize; ++i) {
			for (int j = 0; j < dataSize; ++j) {
				result[i][j] = sumOfProducts[i][j] / dataCount - average[i] * average[j];
			}
		}
		return result;
	}
}
