import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class Main {
	private static List<String> songs = new ArrayList<>();
	private static Map<String, Genre> labels = new HashMap<>();
	private static Map<String, double[]> features = new HashMap<>();

	public static void main(String[] args) throws FileNotFoundException {
		getTrainingSet();
		getTrainingLabels();

		SongClassifier agent = new GaussianClassifier();
		System.out.println("Gaussian: " + crossValidate(agent));

		// SongClassifier agent = new GaussianClassifier();
		// train(agent);
		// classifyTestSet(agent);
	}

	private static void getTrainingSet() throws FileNotFoundException {
		File dir = new File("./training-set/");
		File[] files = dir.listFiles();
		for (int i = 0; i < files.length; ++i) {
			String filename = files[i].getName();
			Stats stats = new Stats();
			Scanner in = new Scanner(new FileInputStream(files[i]));
			while (in.hasNextLine()) {
				String line = in.nextLine();
				String[] data = line.split(",");
				double[] feature = new double[Song.DATA_SIZE];
				for (int j = 0; j < Song.DATA_SIZE; ++j) {
					feature[j] = Double.parseDouble(data[j]);
				}
				stats.add(feature);
			}
			songs.add(filename);
			features.put(filename, stats.average());
			in.close();
		}
	}

	private static void getTrainingLabels() throws FileNotFoundException {
		Scanner in = new Scanner(new FileInputStream("labels.csv"));
		in.nextLine(); // Skip over titles.
		while (in.hasNextLine()) {
			String line = in.nextLine();
			String[] fields = line.split(",");
			labels.put(fields[0], Genre.fromString(fields[1]));
		}
		in.close();
	}

	private static double crossValidate(SongClassifier agent) {
		int n = songs.size();
		double total = 0;
		for (int i = 0; i < 10; ++i) {
			agent.clear();
			int k = n / 10;
			for (int j = 0; j < n; ++j) {
				if (j >= i * k && j < (i + 1) * k)
					continue;
				String name = songs.get(j);
				agent.add(features.get(name), labels.get(name));
			}
			agent.train();
			int correct = 0;
			for (int j = i * k; j < (i + 1) * k; ++j) {
				String name = songs.get(j);
				Genre genre = agent.classify(features.get(name));
				if (genre.equals(labels.get(name)))
					++correct;
			}
			total += (double) (correct) / k;
		}
		return total / 10;
	}

	private static void train(SongClassifier agent) {
		agent.clear();
		int n = songs.size();
		for (int i = 0; i < n; ++i) {
			String name = songs.get(i);
			agent.add(features.get(name), labels.get(name));
		}
		agent.train();
	}

	private static void classifyTestSet(SongClassifier agent) throws FileNotFoundException {
		PrintWriter out = new PrintWriter(new FileOutputStream("results.csv"));
		out.println("id,category");
		File dir = new File("./test-set/");
		File[] files = dir.listFiles();
		for (int i = 0; i < files.length; ++i) {
			String filename = files[i].getName();
			Stats stats = new Stats();
			Scanner in = new Scanner(new FileInputStream(files[i]));
			while (in.hasNextLine()) {
				String line = in.nextLine();
				String[] data = line.split(",");
				double[] feature = new double[Song.DATA_SIZE];
				for (int j = 0; j < Song.DATA_SIZE; ++j) {
					feature[j] = Double.parseDouble(data[j]);
				}
				stats.add(feature);
			}
			Genre genre = agent.classify(stats.average());
			out.println(filename + "," + genre.toString());
			in.close();
		}
		out.close();
	}
}
