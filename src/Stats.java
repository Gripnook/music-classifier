public class Stats {
	private int dataCount = 0;
	private double[] sum = new double[Song.DATA_SIZE];
	private double[][] sumOfProducts = new double[Song.DATA_SIZE][Song.DATA_SIZE];

	public void add(double[] data) {
		++dataCount;
		for (int i = 0; i < Song.DATA_SIZE; ++i) {
			sum[i] += data[i];
			for (int j = 0; j < Song.DATA_SIZE; ++j) {
				sumOfProducts[i][j] += data[i] * data[j];
			}
		}
	}

	public double[] average() {
		if (dataCount == 0)
			return null;

		double[] result = new double[Song.DATA_SIZE];
		for (int i = 0; i < Song.DATA_SIZE; ++i) {
			result[i] = sum[i] / dataCount;
		}
		return result;
	}

	public double[][] covariance() {
		if (dataCount == 0)
			return null;

		double[] average = average();
		double[][] result = new double[Song.DATA_SIZE][Song.DATA_SIZE];
		for (int i = 0; i < Song.DATA_SIZE; ++i) {
			for (int j = 0; j < Song.DATA_SIZE; ++j) {
				result[i][j] = sumOfProducts[i][j] / dataCount - average[i] * average[j];
			}
		}
		return result;
	}
}
