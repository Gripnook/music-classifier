package numeric;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.ejml.data.Complex_F64;
import org.ejml.simple.SimpleEVD;
import org.ejml.simple.SimpleMatrix;

public class PCA {
	private SimpleMatrix transform;

	public PCA(double[][] covariance, int targetSize) {
		SimpleEVD<SimpleMatrix> evd = Matrix.convertMatrix(covariance).eig();
		List<Complex_F64> eigenvalues = evd.getEigenvalues();
		List<Integer> indices = new ArrayList<>();
		for (int i = 0; i < eigenvalues.size(); ++i) {
			indices.add(i);
		}
		Collections.sort(indices, (lhs, rhs) -> -Double.compare(eigenvalues.get(lhs).real, eigenvalues.get(rhs).real));

		int originalSize = eigenvalues.size();
		transform = new SimpleMatrix(targetSize, originalSize);
		for (int i = 0; i < targetSize; ++i) {
			SimpleMatrix eigenvector = evd.getEigenVector(indices.get(i));
			double norm = eigenvector.normF();
			for (int j = 0; j < originalSize; ++j) {
				transform.set(i, j, eigenvector.get(j) / norm);
			}
		}
	}

	public double[] transform(double[] feature) {
		return Matrix.convertVector(transform.mult(Matrix.convertVector(feature)));
	}
}
