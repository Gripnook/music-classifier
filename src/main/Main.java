package main;

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

import classifier.DecisionForestClassifier;
import classifier.GaussianClassifier;
import classifier.KNNClassifier;
import classifier.SongClassifier;
import classifier.TotalGaussianClassifier;
import classifier.WeighedKNNClassifier;

public class Main {
	private static List<String> songNames = new ArrayList<>();
	private static Map<String, Genre> labels = new HashMap<>();
	private static Map<String, List<double[]>> songs = new HashMap<>();

	public static void main(String[] args) throws FileNotFoundException {
		getTrainingSet();
		getTrainingLabels();

		SongClassifier agent;

		agent = new GaussianClassifier();
		System.out.println("gaussian: " + leaveOneOutCrossValidate(agent));
		agent = new TotalGaussianClassifier();
		System.out.println("total gaussian: " + leaveOneOutCrossValidate(agent));
		agent = new KNNClassifier(1);
		System.out.println("1NN: " + leaveOneOutCrossValidate(agent));
		agent = new KNNClassifier(10);
		System.out.println("10NN: " + leaveOneOutCrossValidate(agent));
		agent = new WeighedKNNClassifier(50);
		System.out.println("weighed 50NN: " + leaveOneOutCrossValidate(agent));
		agent = new DecisionForestClassifier(10);
		System.out.println("decision forest (10): " + leaveOneOutCrossValidate(agent));

		// train(agent);
		// classifyTestSet(agent);
	}

	private static void getTrainingSet() throws FileNotFoundException {
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

	private static double crossValidate(SongClassifier agent, int testSets) {
		int n = songNames.size();
		int testSetSize = n / testSets;
		double total = 0;
		for (int i = 0; i < testSets; ++i) {
			System.out.print((i + 1) + " / " + testSets + "; ");
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
		}
		System.out.println();
		return total / testSets;
	}

	private static double leaveOneOutCrossValidate(SongClassifier agent) {
		return crossValidate(agent, songNames.size());
	}

	private static void train(SongClassifier agent) {
		agent.clear();
		int n = songNames.size();
		for (int i = 0; i < n; ++i) {
			String name = songNames.get(i);
			agent.add(songs.get(name), labels.get(name));
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
	}
}
