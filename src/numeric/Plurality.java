package numeric;

import java.util.HashMap;
import java.util.Map;

/**
 * A helper class that keeps track of how many times an object appears in the
 * data and performs plurality voting on the object most seen.
 * 
 * @author Andrei Purcarus
 *
 * @param <T>
 */
public class Plurality<T> {
	private Map<T, Integer> counts = new HashMap<>();

	/**
	 * Adds the object to the data set.
	 * 
	 * @param object
	 */
	public void add(T object) {
		Integer count = counts.get(object);
		if (count == null) {
			count = 0;
		}
		++count;
		counts.put(object, count);
	}

	/**
	 * Gets the plurality vote of the current data set.
	 * 
	 * @return
	 */
	public T vote() {
		int max = 0;
		T result = null;
		for (T object : counts.keySet()) {
			int count = counts.get(object);
			if (count > max) {
				max = count;
				result = object;
			}
		}
		return result;
	}
}
