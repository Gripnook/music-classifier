package numeric;

import java.util.HashMap;
import java.util.Map;

public class Plurality<T> {
	private Map<T, Integer> counts = new HashMap<>();

	public void add(T object) {
		Integer count = counts.get(object);
		if (count == null) {
			count = 0;
		}
		++count;
		counts.put(object, count);
	}

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
