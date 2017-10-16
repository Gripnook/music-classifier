package numeric;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.ejml.data.Complex_F64;
import org.ejml.simple.SimpleEVD;
import org.ejml.simple.SimpleMatrix;

/**
 * A transformation class that performs Primary Component Analysis on a set of
 * data. It can be used to reduce the dimensionality of feature sets or simply
 * to extract the primary components with the same dimensionality.
 * 
 * @author Andrei Purcarus
 *
 */
public class PCA {
	private SimpleMatrix transform;

	/**
	 * Creates a transformation object that maps data with the given covariance
	 * matrix into data of the target size.
	 * 
	 * @param covariance
	 * @param targetSize
	 */
	public PCA(double[][] covariance, int targetSize) {
		// Gets the eigenvectors of the covariance matrix and sorts them by
		// corresponding eigenvalue, largest first.
		SimpleEVD<SimpleMatrix> evd = Matrix.convertMatrix(covariance).eig();
		List<Complex_F64> eigenvalues = evd.getEigenvalues();
		List<Integer> indices = new ArrayList<>();
		for (int i = 0; i < eigenvalues.size(); ++i) {
			indices.add(i);
		}
		Collections.sort(indices, (lhs, rhs) -> -Double.compare(eigenvalues.get(lhs).real, eigenvalues.get(rhs).real));

		// Creates the transformation matrix T by normalizing the eigenvectors
		// and using them as rows of T.
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

	/**
	 * Transforms the feature vector into the target size using PCA.
	 * 
	 * @param feature
	 * @return
	 */
	public double[] transform(double[] feature) {
		return Matrix.convertVector(transform.mult(Matrix.convertVector(feature)));
	}
}
