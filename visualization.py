import plotly.graph_objects as go

# If using Jupyter notebook, use the following line to render the plot in the notebook correctly
import plotly.io as pio

pio.renderers.default = "iframe"


def visualize_paragraph(
    paragraph, word_metrics, width=5000, scaling_factor=0.1, ent_thres=2, vare_thres=1
):
    """
    Visualize a paragraph of text with hover metrics for each word using Plotly.

    :param paragraph: A list of word
    :param word_metrics: A list metrics for word
    :param max_line_length: Maximum line length for text wrapping.
    :param scaling_factor: Scaling factor for x positions.
    """

    words = []
    xs = []
    ys = []
    hover_texts = []

    y = 0
    x = 0  # Initialize x position for the line
    shapes = []
    for word, metric in zip(paragraph, word_metrics):
        if word == "\n":
            x = 0
            y += 1
            continue
        words.append(word)
        xs.append(x)
        ys.append((-y) * 0.1)  # Negative y to have lines from top to bottom
        x += (len(word) + 1) * scaling_factor  # Adjust scaling factor as needed
        # Get the metric for the word
        hover_texts.append(f"{word}: {metric}")
        # If metric is high, add a rectangle shape behind the word
        ent = float(metric.split(",")[0])
        vare = float(metric.split(",")[1])
        if ent > ent_thres:
            word_length = len(word)
            x0 = xs[-1] - scaling_factor * 0.2
            x1 = xs[-1] + (word_length + 0.2) * scaling_factor
            y0 = ys[-1] - 0.01  # Adjust height as needed
            y1 = ys[-1] + 0.01
            shape = dict(
                type="rect",
                xref="x",
                yref="y",
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                fillcolor=f"rgba(255, 0, 0, {.5})",  # Red for high entropy words
                line=dict(width=0),
                layer="below",
            )
            shapes.append(shape)
        if vare > vare_thres:
            word_length = len(word)
            x0 = xs[-1] - scaling_factor * 0.2
            x1 = xs[-1] + (word_length + 0.2) * scaling_factor
            y0 = ys[-1] - 0.01  # Adjust height as needed
            y1 = ys[-1] + 0.01
            shape = dict(
                type="rect",
                xref="x",
                yref="y",
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                fillcolor=f"rgba(0, 255, 0, {.5})",  # Green for high varentropy words
                line=dict(width=0),
                layer="below",
            )
            shapes.append(shape)
    # print([(x, y) for x, y in zip(xs, ys)])
    # print(words)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="text",
            text=words,
            textfont=dict(family="Arial", size=14, color="black"),
            textposition="middle right",
            hovertext=hover_texts,
            hoverinfo="text",
        )
    )

    fig.update_xaxes(visible=True)
    fig.update_yaxes(visible=True)
    fig.update_layout(
        shapes=shapes,
        showlegend=False,
        hovermode="closest",
        autosize=False,
        margin=dict(l=10, r=10, t=10, b=10),
        width=width,
    )

    fig.show()
