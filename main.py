import click
import numpy as np

from attack_graph import MulvalAttackGraph, AttackGraph
from attack_graph_generation import Generator
from clustering.white_smyth import Spectral1, Spectral2
from embedding.deepwalk import DeepWalk
from embedding.embedding import Embedding
from embedding.graphsage import Graphsage
from embedding.hope import Hope
from ranking.mehta import PageRankMethod, KuehlmannMethod


@click.group()
def run():
    pass


@run.command()
@click.option(
    "--keep_mulval",
    is_flag=True,
    help="Whether or not the attack graph should be kept as a MulVAL attack"
    " graph and not converted to a standard attack graph.")
@click.option(
    "-r",
    "--ranking",
    type=click.Choice(["pagerank", "kuehlmann"]),
    help="Indicates if a ranking method should be applied and which method to"
    " apply.")
@click.option(
    "--cluster",
    type=click.Choice(["spectral1", "spectral2", "embedding"]),
    help="Indicates if a clustering method should be applied and which method"
    " to apply.")
@click.option(
    "--embedding",
    type=click.Path(exists=True),
    help="Indicates the path to an embedding file. The embedding can then be"
    " used for clustering.")
@click.argument("input", required=True)
@click.argument("output", required=True)
def draw(input: str, output: str, keep_mulval: bool, ranking: str,
         cluster: str, embedding: str):
    """
    Draws the attack graph xml file located at INPUT to OUTPUT.

    INPUT is the location of the attack graph file. The extension of the path
    should be in [xml|gml]
    OUTPUT is the location where the attack graph should be drawn. The
    extension of the path should be in [dot|pdf|png].
    """
    # Create the attack graph
    if keep_mulval:
        ag = MulvalAttackGraph()
    else:
        ag = AttackGraph()

    ag.load(input)

    # Apply ranking
    if ranking == "pagerank":
        PageRankMethod(ag).apply()
    elif ranking == "kuehlmann":
        KuehlmannMethod(ag).apply()

    # Apply clustering
    if cluster == "spectral1":
        Spectral1(ag).apply(5)
    elif cluster == "spectral2":
        Spectral2(ag).apply(1, 9)
    elif cluster == "embedding":
        user_embedding = np.load(embedding)
        emb = Embedding(ag, user_embedding.shape[1])
        emb.embedding = user_embedding
        emb.cluster()

    # Draw the resulting graph
    ag.draw(output)


@run.command()
@click.option("-p",
              "--n_propositions",
              type=click.INT,
              help="The total number of propositions.",
              default=20)
@click.option("-i",
              "--n_initial_propositions",
              type=click.INT,
              help="The number of propositions that are initially true.",
              default=10)
@click.option("-e",
              "--n_exploits",
              type=click.INT,
              help="The number of exploits that should be generated.",
              default=20)
@click.argument("output", required=True)
def generate(output: str,
             n_propositions: int = 20,
             n_initial_propositions: int = 10,
             n_exploits: int = 20):
    """
    Generates a random attack graph.

    OUTPUT is the location where the attack graph should be saved.
    """
    ag = Generator(n_propositions, n_initial_propositions,
                   n_exploits).generate()
    ag.save(output)


@run.command()
@click.option("-d",
              "--dim_embedding",
              type=click.Choice(["8", "16", "32", "64"]),
              help="The dimension of the embedding.",
              default="8")
@click.argument("input", required=True)
def deepwalk(input: str, dim_embedding: str = "8"):
    """
    Applies Deepwalk to the attack graph located at INPUT. Saves the embedding
    at methods_output/deepwalk/embeddings.npy in a pickle file.

    INPUT is the location of the attack graph file. The extension of the path
    should be in [xml|gml]
    """
    # Create the attack graph
    ag = AttackGraph()
    ag.load(input)

    # Apply DeepWalk
    DeepWalk(ag, int(dim_embedding)).run()


@run.command()
@click.option("-d",
              "--dim_embedding",
              type=click.Choice(["8", "16", "32", "64"]),
              help="The dimension of the embedding.",
              default="8")
@click.option(
    "-m",
    "--measurement",
    type=click.Choice(["cn", "katz", "pagerank", "aa"]),
    help="The measurement to use. Either cn (Common Neighbours),"
    " katz (Katz), pagerank (Personalized Pagerank) or aa (Adamic-Adar)",
    default="cn")
@click.argument("input", required=True)
def hope(input: str, dim_embedding: str = "8", measurement: str = "cn"):
    """
    Applies HOPE to the attack graph located at INPUT. Saves the embedding
    at methods_output/hope/embeddings.npy in a pickle file.

    INPUT is the location of the attack graph file. The extension of the path
    should be in [xml|gml].
    """
    # Create the attack graph
    ag = AttackGraph()
    ag.load(input)

    # Apply Hope
    Hope(ag, int(dim_embedding), measurement).run()


@run.command()
@click.option("-d",
              "--dim_embedding",
              type=click.Choice(["8", "16", "32", "64"]),
              help="The dimension of the embedding.",
              default="8")
@click.argument("input", required=True)
def graphsage(input: str, dim_embedding: str = "8"):
    """
    Applies GraphSAGE to the attack graph located at INPUT. Saves the embedding
    at methods_output/graphsage/embeddings.npy in a pickle file.

    INPUT is the location of the attack graph file. The extension of the path
    should be in [xml|gml]
    """
    # Create the attack graph
    ag = AttackGraph()
    ag.load(input)

    # Apply GraphSAGE
    Graphsage(ag, int(dim_embedding), "toy").run()


if __name__ == "__main__":
    run()
