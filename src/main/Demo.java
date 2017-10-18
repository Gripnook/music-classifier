package main;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import classifier.SongClassifier;
import classifier.gaussian.GaussianClassifier;
import classifier.gaussian.TotalGaussianClassifier;
import classifier.knn.KNNClassifier;
import classifier.knn.WeighedKNNClassifier;
import classifier.tree.DecisionForestClassifier;

public class Demo {
	private static List<String> songNames = new ArrayList<>();
	private static Map<String, Genre> labels = new HashMap<>();
	private static Map<String, List<double[]>> songs = new HashMap<>();

	public static void main(String[] args) throws FileNotFoundException {
		getTrainingSet();
		getTrainingLabels();

		Scanner in = new Scanner(System.in);
		while (true) {
			SongClassifier agent = getAgent(in);
			performAction(in, agent);
		}
	}

	private static void getTrainingSet() throws FileNotFoundException {
		System.out.print("getting training data... ");
		File dir = new File("./training-set/");
		File[] files = dir.listFiles();
		for (int i = 0; i < files.length; ++i) {
			String filename = files[i].getName();
			List<double[]> song = new ArrayList<>();
			Scanner in = new Scanner(new FileInputStream(files[i]));
			while (in.hasNextLine()) {
				String line = in.nextLine();
				String[] data = line.split(",");
				double[] feature = new double[Song.FEATURES];
				for (int j = 0; j < Song.FEATURES; ++j) {
					feature[j] = Double.parseDouble(data[j]);
				}
				song.add(feature);
			}
			songNames.add(filename);
			songs.put(filename, song);
			in.close();
		}
		System.out.println("done");
	}

	private static void getTrainingLabels() throws FileNotFoundException {
		System.out.print("getting training labels... ");
		Scanner in = new Scanner(new FileInputStream("labels.csv"));
		in.nextLine(); // Skip over titles.
		while (in.hasNextLine()) {
			String line = in.nextLine();
			String[] fields = line.split(",");
			labels.put(fields[0], Genre.fromString(fields[1]));
		}
		in.close();
		System.out.println("done");
	}

	private static SongClassifier getAgent(Scanner in) {
		SongClassifier agent = null;
		while (true) {
			System.out.println("Select an agent:");
			System.out.println("(1) gaussian");
			System.out.println("(2) total gaussian");
			System.out.println("(3) kNN");
			System.out.println("(4) weighed kNN");
			System.out.println("(5) random forest");
			while (!in.hasNextInt()) {
				in.nextLine();
				System.out.println("invalid selection");
				continue;
			}
			int choice = in.nextInt();
			if (choice == 1) {
				agent = new GaussianClassifier();
				break;
			} else if (choice == 2) {
				agent = new TotalGaussianClassifier();
				break;
			} else if (choice == 3) {
				int k = getInteger(in, "what value of k? ", 1);
				agent = new KNNClassifier(k);
				break;
			} else if (choice == 4) {
				int k = getInteger(in, "what value of k? ", 1);
				agent = new WeighedKNNClassifier(k);
				break;
			} else if (choice == 5) {
				int numTrees = getInteger(in, "how many trees in the forest? ", 1);
				agent = new DecisionForestClassifier(numTrees);
				break;
			} else {
				in.nextLine();
				System.out.println("invalid selection");
				continue;
			}
		}
		return agent;
	}

	private static void performAction(Scanner in, SongClassifier agent) throws FileNotFoundException {
		while (true) {
			System.out.println("select an action to perform:");
			System.out.println("(1) cross-validation");
			System.out.println("(2) test set classification");
			while (!in.hasNextInt()) {
				in.nextLine();
				System.out.println("invalid selection");
				continue;
			}
			int choice = in.nextInt();
			if (choice == 1) {
				int testSets = getInteger(in, "how many sets to partition the data into? ", 2);
				System.out.println("cross validating...");
				System.out.println(
						"results for " + agent.getClass().getSimpleName() + ": " + crossValidate(agent, testSets));
				return;
			} else if (choice == 2) {
				train(agent);
				classifyTestSet(agent);
				return;
			} else {
				in.nextLine();
				System.out.println("invalid selection");
				continue;
			}
		}
	}

	private static int getInteger(Scanner in, String prompt, int min) {
		while (true) {
			System.out.print(prompt);
			while (!in.hasNextInt()) {
				in.nextLine();
				System.out.println("invalid selection");
				continue;
			}
			int result = in.nextInt();
			if (result < min) {
				in.nextLine();
				System.out.println("invalid selection");
				continue;
			}
			return result;
		}
	}

	private static double crossValidate(SongClassifier agent, int testSets) {
		int n = songNames.size();
		int testSetSize = n / testSets;
		double total = 0;
		Collections.shuffle(songNames);
		for (int i = 0; i < testSets; ++i) {
			System.out.print((i + 1) + " / " + testSets + ": ");
			agent.clear();
			for (int j = 0; j < n; ++j) {
				if (j >= i * testSetSize && j < (i + 1) * testSetSize) {
					continue;
				}
				String name = songNames.get(j);
				agent.add(songs.get(name), labels.get(name));
			}
			agent.train();
			int correct = 0;
			for (int j = i * testSetSize; j < (i + 1) * testSetSize; ++j) {
				String name = songNames.get(j);
				Genre genre = agent.classify(songs.get(name));
				if (genre.equals(labels.get(name))) {
					++correct;
				}
			}
			total += (double) (correct) / testSetSize;
			System.out.print(((double) (correct) / testSetSize) + "; ");
		}
		System.out.println();
		return total / testSets;
	}

	private static void train(SongClassifier agent) {
		System.out.print("training... ");
		agent.clear();
		int n = songNames.size();
		for (int i = 0; i < n; ++i) {
			String name = songNames.get(i);
			agent.add(songs.get(name), labels.get(name));
		}
		agent.train();
		System.out.println("done");
	}

	private static void classifyTestSet(SongClassifier agent) throws FileNotFoundException {
		System.out.print("classifying test data... ");
		String outName = "results." + agent.getClass().getSimpleName().toLowerCase() + ".csv";
		PrintWriter out = new PrintWriter(new FileOutputStream(outName));
		out.println("id,category");
		File dir = new File("./test-set/");
		File[] files = dir.listFiles();
		for (int i = 0; i < files.length; ++i) {
			String filename = files[i].getName();
			List<double[]> song = new ArrayList<>();
			Scanner in = new Scanner(new FileInputStream(files[i]));
			while (in.hasNextLine()) {
				String line = in.nextLine();
				String[] data = line.split(",");
				double[] feature = new double[Song.FEATURES];
				for (int j = 0; j < Song.FEATURES; ++j) {
					feature[j] = Double.parseDouble(data[j]);
				}
				song.add(feature);
			}
			Genre genre = agent.classify(song);
			out.println(filename + "," + genre.toString());
			in.close();
		}
		out.close();
		System.out.println("done");
		System.out.println("results have been output to: " + outName);
	}
}
