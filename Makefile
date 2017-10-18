all:
	mkdir -p build
	javac -d build -classpath 'ejml-v0.32-libs/*' src/*/*.java src/*/*/*.java
	java -classpath 'build:ejml-v0.32-libs/*' main.Demo
