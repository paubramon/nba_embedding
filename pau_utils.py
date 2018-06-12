import pandas as pd
import os,errno


def export_latex_table(table, filename="table", path="./", latex_prefix="", latex_suffix="", caption_table =""):
    """
    Function to create a latex table with some suffix or prefix. This is basically used to add a scalebox and a caption
    and label. However, you can also directly specify the prefix or suffix as well as the caption table. Note that if
    you add prefix and suffix caption_table will be overwritten. Note that adding prefix and suffix can lead to latex
    errors, since there can be incoherences in their definition.
    :param table: this is the pandas Dataframe defining the table
    :param filename: this is the name of the .tex file. Note that you don't have to add the extension .tex.
    :param path: this is the path where you want to save your table
    :param latex_prefix: optional, to define a custom prefix
    :param latex_suffix: optional, to define a custom suffix
    :param caption_table: optional, to define a custom caption.
    :return:
    """
    if caption_table:
        caption = caption_table
    else:
        caption = "Table " + filename

    if not latex_prefix and not latex_suffix:
        prefix = """\\begin{table}[H]
\\begin{center}
\scalebox{0.6}{"""
        suffix = """}
\end{center}
\caption{""" + caption.replace("_"," ") + """}
\label{tab:table_""" + filename + """}
\end{table}"""

    else:
        prefix = latex_prefix
        suffix = latex_suffix

    latex_table = prefix + table.to_latex() + suffix

    with open(path + filename + '.tex', 'w') as tex_file:
        tex_file.write(latex_table)


def create_folder(directory):
    """
    Check if the directory exists and creates it otherwise.
    :param path:
    :return:
    """

    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise