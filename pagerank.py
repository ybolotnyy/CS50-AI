import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # Initialize a dictionary to store the transition probabilities
    transition_probabilities = {}

    # Total number of pages in the corpus
    total_pages = len(corpus)

    # Check if the current page has outgoing links
    if not corpus[page]:
        # Handle the case where there are no outgoing links
        # Assign equal probability to all pages, including itself
        probability = 1 / total_pages
        for p in corpus:
            transition_probabilities[p] = probability
    else:
        # Calculate the probability of following a link from the current page
        link_probability = (1 - damping_factor) / total_pages

        # Calculate the probability of choosing a link from the current page
        for p in corpus:
            transition_probabilities[p] = link_probability

        # Calculate the probability of choosing any page at random
        random_probability = damping_factor / len(corpus[page])

        # Adjust the probabilities for pages linked to the current page
        for p in corpus[page]:
            transition_probabilities[p] += random_probability

    return transition_probabilities


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Initialize a dictionary to store PageRank estimates
    pagerank = {page: 0 for page in corpus}

    # Start with a random page
    page = random.choice(list(corpus.keys()))

    for _ in range(n):
        # Generate a transition model for the current page
        transition_probabilities = transition_model(corpus, page, damping_factor)

        # Choose the next page based on the transition model
        page = random.choices(list(transition_probabilities.keys()), weights=list(transition_probabilities.values()))[0]

        # Update the count for the chosen page
        pagerank[page] += 1

    # Normalize the counts to get PageRank estimates
    total_samples = sum(pagerank.values())
    for page in pagerank:
        pagerank[page] /= total_samples

    return pagerank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Initialize PageRank values for each page
    n = len(corpus)
    pagerank = {page: 1 / n for page in corpus}

    # Convergence threshold
    threshold = 0.001

    while True:
        new_pagerank = {}
        for page in corpus:
            new_rank = (1 - damping_factor) / n  # Start with the probability to jump to any page
            for p in corpus:
                if page in corpus[p]:
                    new_rank += damping_factor * pagerank[p] / len(corpus[p])
            new_pagerank[page] = new_rank

        # Check for convergence
        max_change = max(abs(new_pagerank[page] - pagerank[page]) for page in pagerank)
        if max_change < threshold:
            return new_pagerank

        pagerank = new_pagerank


if __name__ == "__main__":
    main()
