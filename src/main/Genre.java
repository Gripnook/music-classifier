package main;

public enum Genre {
	classical("classical"), country("country"), edm_dance("edm_dance"), jazz("jazz"), kids("kids"), latin(
			"latin"), metal("metal"), pop("pop"), rnb("rnb"), rock("rock");

	public static Genre fromString(String name) {
		for (Genre genre : Genre.class.getEnumConstants()) {
			if (genre.toString().equals(name)) {
				return genre;
			}
		}
		throw new Error("invalid genre");
	}

	@Override
	public String toString() {
		return name;
	}

	private String name;

	private Genre(String name) {
		this.name = name;
	}
}
