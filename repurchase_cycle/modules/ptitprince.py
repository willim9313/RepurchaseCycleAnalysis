"""
Raincloud Plot Visualization Utilities
======================================

This file is adapted from the PtitPrince project:
https://github.com/pog87/PtitPrince

Original author: Davide Poggiali
License: BSD 3-Clause License

Modifications have been made for:
- Compatibility with seaborn 0.13+ API changes
- Project-specific customization and styling
- Integration with the repurchase cycle analysis pipeline
"""

import warnings
from colorsys import rgb_to_hls
from typing import Any, Optional, Union

import matplotlib as mpl
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from seaborn.categorical import *
from seaborn.categorical import _CategoricalPlotter  # , _CategoricalScatterPlotter

__all__ = ["half_violinplot", "stripplot", "RainCloud"]
__version__ = "0.3.1"

# Define a type alias for data inputs for reusability
DataInput = Optional[Union[pd.Series, np.ndarray, list]]

# Legacy gray color constant for backward compatibility
_LEGACY_GRAY = "0.3"  # Matches the gray color from older seaborn versions


def _get_complementary_gray(base_color, hue_map):
    """Creates a gray color that complements the plot's main colors."""
    if hue_map.lookup_table is None:
        if base_color is None:
            return "0.3"  # A safe default
        basis = [mpl.colors.to_rgb(base_color)]
    else:
        basis = [mpl.colors.to_rgb(c) for c in hue_map.lookup_table.values()]

    unique_colors = np.unique(basis, axis=0)
    light_vals = [rgb_to_hls(*rgb[:3])[1] for rgb in unique_colors]
    lum = min(light_vals) * 0.6
    return (lum, lum, lum)


class _Half_ViolinPlotter(_CategoricalPlotter):
    def __init__(
        self,
        x,
        y,
        hue,
        data,
        order,
        hue_order,
        bw,
        cut,
        scale,
        scale_hue,
        gridsize,
        width,
        inner,
        split,
        dodge,
        orient,
        linewidth,
        color,
        palette,
        saturation,
        offset,
    ):
        variables = dict(x=x, y=y, hue=hue)

        super().__init__(
            data=data,
            variables=variables,
            order=order,
            orient=orient,
            color=color,
        )

        # Set attributes expected by parent class methods
        # _redundant_hue is False when hue is a distinct semantic variable
        self._redundant_hue = False
        self.legend = "auto"  # Use automatic legend behavior

        # Store the palette for later use when there's no hue
        # In seaborn 0.13, palette is ignored if there's no hue variable
        self._user_palette = palette
        self._user_color = color

        # Only pass palette to map_hue if there's a hue variable to avoid warning
        if hue is not None:
            self.map_hue(palette=palette, order=hue_order, saturation=saturation)
        else:
            self.map_hue(palette=None, order=hue_order, saturation=saturation)

        self.bw = bw
        self.cut = cut
        self.scale = scale
        self.scale_hue = scale_hue
        self.gridsize = gridsize
        self.width = width
        self.dodge = dodge
        self.offset = offset

        if inner is not None:
            if not any(
                [
                    inner.startswith("quart"),
                    inner.startswith("box"),
                    inner.startswith("stick"),
                    inner.startswith("point"),
                ]
            ):
                err = f"Inner style '{inner}' not recognized"
                raise ValueError(err)
        self.inner = inner

        if split and "hue" in self.variables and len(self.var_levels.get("hue", [])) < 2:
            msg = "There must be at least two hue levels to use `split`.'"
            raise ValueError(msg)
        self.split = split

        if linewidth is None:
            linewidth = mpl.rcParams["lines.linewidth"]
        self.linewidth = linewidth

        self.gray = _get_complementary_gray(color, self._hue_map)
        # Fallback to explicit gray if complement color is not working
        if self.gray is None or self.gray == "none":
            self.gray = "0.3"

    def estimate_densities(self, bw, cut, scale, scale_hue, gridsize):
        """Find the support and density for all of the data."""
        # Initialize data structures to keep track of plotting data
        violin_data = []

        # In the modern seaborn structure, the orient might be the actual axis name
        # Let's determine based on which variable is categorical vs numeric
        if "x" in self.variables and "y" in self.variables:
            # Determine orientation: for vertical plots x=categorical, y=numeric
            # For horizontal plots x=numeric, y=categorical
            # Note: modern seaborn sets self.orient to 'y' for horizontal plots, 'x' for vertical
            if self.orient == "y":
                # Horizontal: x=numeric (values), y=categorical (groups)
                value_variable = "x"
                categorical_variable = "y"
            else:
                # Vertical: x=categorical (groups), y=numeric (values)
                value_variable = "y"
                categorical_variable = "x"
        else:
            # Fallback to original logic
            value_variable = "y" if self.orient == "v" else "x"
            categorical_variable = "x" if self.orient == "v" else "y"
        grouping_vars = [categorical_variable]  # e.g., ['x']
        if "hue" in self.variables:
            grouping_vars.append("hue")

        for group_name, group_df in self.plot_data.groupby(grouping_vars):
            # Extract numeric values and remove NaN values
            values = group_df[value_variable]
            kde_data = values.dropna()

            # Handle edge cases for this specific violin
            if kde_data.size == 0:
                support_i = np.array([])
                density_i = np.array([1.0])
            elif np.unique(kde_data).size == 1:
                support_i = np.unique(kde_data)
                density_i = np.array([1.0])
            else:
                # Fit the KDE for this violin's data
                kde, bw_used = self.fit_kde(kde_data, bw)
                support_i = self.kde_support(kde_data, bw_used, cut, gridsize)
                density_i = kde.evaluate(support_i)

            # 4. Store all results for this one violin in a dictionary
            violin_data.append(
                {
                    "group_name": group_name,
                    "support": support_i,
                    "density": density_i,
                    "observations": kde_data,
                    "max_density": density_i.max() if density_i.size > 1 else 0,
                    "count": kde_data.size,
                }
            )

        # 5. Store the complete results list first
        self.violin_data = violin_data

        # 6. Apply scaling
        if scale == "area":
            self.scale_area(scale_hue)

        elif scale == "width":
            self._scale_width()

        elif scale == "count":
            self.scale_count(scale_hue)

        else:
            raise ValueError(f"scale method '{scale}' not recognized")

    def fit_kde(self, x, bw):
        """Estimate a KDE for a vector of data with flexible bandwidth."""
        # Ensure x is a numpy array of floats
        x = np.asarray(x, dtype=float)

        # Allow for the use of old scipy where `bw` is fixed
        try:
            kde = stats.gaussian_kde(x, bw)
        except TypeError:
            kde = stats.gaussian_kde(x)
            if bw != "scott":  # scipy default
                msg = (
                    "Ignoring bandwidth choice, please upgrade scipy to use a different bandwidth."
                )
                warnings.warn(msg, UserWarning)

        # Extract the numeric bandwidth from the KDE object
        bw_used = kde.factor

        # At this point, bw will be a numeric scale factor.
        # To get the actual bandwidth of the kernel, we multiple by the
        # unbiased standard deviation of the data, which we will use
        # elsewhere to compute the range of the support.
        bw_used = bw_used * x.std(ddof=1)

        return kde, bw_used

    def kde_support(self, x, bw, cut, gridsize):
        """Define a grid of support for the violin."""
        support_min = x.min() - bw * cut
        support_max = x.max() + bw * cut
        return np.linspace(support_min, support_max, gridsize)

    def scale_area(self, scale_hue):
        """Scale the densities in self.violin_data to preserve area."""

        # First, find the overall maximum density if we need it
        global_max_density = 1
        if not scale_hue:
            # Get a list of all max_densities and find the true global maximum
            all_max_densities = [d["max_density"] for d in self.violin_data]
            if all_max_densities:
                global_max_density = max(all_max_densities)

        # If scaling by hue, we need to find the max density within each category
        if "hue" in self.variables and scale_hue:
            # Use pandas to quickly find the max density for each x-category
            df = pd.DataFrame(self.violin_data)
            # The group_name is a tuple like ('Sad', 'Friend'), self.orient is 'x'
            df["orient_cat"] = [
                name[0] if isinstance(name, tuple) else name for name in df["group_name"]
            ]
            category_maxes = df.groupby("orient_cat")["max_density"].transform("max")

        # Now, loop through the list of dictionaries and update the density
        for i, violin in enumerate(self.violin_data):
            density = violin["density"]
            if density.size <= 1:
                continue

            if "hue" not in self.variables:
                # Case 1: No hue, scale by the global max
                scaler = max([d["max_density"] for d in self.violin_data])
            elif scale_hue:
                # Case 2: Hue is present, scale within each x-category
                scaler = category_maxes[i]
            else:
                # Case 3: Hue is present, but scale by the global max
                scaler = global_max_density

            if scaler > 0:
                violin["density"] /= scaler

    def scale_count(self, scale_hue):
        """Scale each density curve by observation count in self.violin_data."""

        # --- 1. Find the maximum counts needed for scaling ---

        # Get a list of all counts to find the global maximum
        all_counts = [d["count"] for d in self.violin_data]
        global_max_count = max(all_counts) if all_counts else 1

        # If scaling by hue, find the max count within each primary category
        if "hue" in self.variables and scale_hue:
            df = pd.DataFrame(self.violin_data)
            # The group_name can be a tuple like ('Sad', 'Friend') or just 'Sad'
            # We extract the first element to get the primary category
            df["orient_cat"] = [
                name[0] if isinstance(name, tuple) else name for name in df["group_name"]
            ]
            # Use pandas `transform` to get the max count for each violin's category
            category_max_counts = df.groupby("orient_cat")["count"].transform("max")

        # --- 2. Loop through violins and apply the scaling ---

        for i, violin in enumerate(self.violin_data):
            density = violin["density"]
            max_density = violin["max_density"]
            count = violin["count"]

            # First, normalize the violin to its own max height of 1
            if max_density > 0:
                normalized_density = density / max_density
            else:
                normalized_density = density

            # Next, determine the scaler based on observation counts
            if "hue" not in self.variables:
                # Case 1: No hue, scale by the global max count
                scaler = count / global_max_count
            elif scale_hue:
                # Case 2: Hue is present, scale within each x-category
                max_count_in_category = category_max_counts[i]
                scaler = count / max_count_in_category if max_count_in_category > 0 else 0
            else:
                # Case 3: Hue is present, but scale by the global max count
                scaler = count / global_max_count

            # Apply the final scaling
            violin["density"] = normalized_density * scaler

    def _scale_width(self):
        """Scale each density curve to the same height in self.violin_data."""
        for violin in self.violin_data:
            density = violin["density"]
            max_density = violin["max_density"]

            # Normalize the density by its own maximum to make the new max 1
            if max_density > 0:
                violin["density"] = density / max_density

    def draw_violins(self, ax, kws):
        """Draw the violins onto `ax`."""
        # Determine correct orientation based on variable types
        # For vertical plots: categorical on x-axis, numeric on y-axis -> use fill_betweenx
        # For horizontal plots: numeric on x-axis, categorical on y-axis -> use fill_between
        if "x" in self.variables and "y" in self.variables:
            if self.orient == "y":
                # Horizontal: x=numeric, y=categorical -> use fill_between
                fill_func = ax.fill_between
            else:
                # Vertical: x=categorical, y=numeric -> use fill_betweenx
                fill_func = ax.fill_betweenx
        else:
            # Fallback to original logic
            fill_func = ax.fill_betweenx if self.orient == "v" else ax.fill_between

        # Set up default drawing properties
        kws.update(dict(edgecolor=self.gray, linewidth=self.linewidth))

        # Calculate width for violin drawing
        if not hasattr(self, "dwidth"):
            self.dwidth = self.width / 2

        # Single loop through violin_data - handles both hue and no-hue cases
        for violin in self.violin_data:
            support = violin["support"]
            density = violin["density"]
            group_name = violin["group_name"]

            # Handle special case of no observations
            if support.size == 0:
                continue

            # Handle special case of a single observation
            if support.size == 1:
                val = np.ndarray.item(support)
                d = np.ndarray.item(density)
                center = self._get_center_position(group_name)
                if self.split and "hue" in self.variables:
                    d = d / 2
                self.draw_single_observation(ax, center, val, d)
                continue

            # Get center position and color for this violin
            center = self._get_center_position(group_name)
            color = self._get_violin_color(group_name)

            # Draw the violin polygon
            grid = np.ones(len(support)) * center

            if self.split and "hue" in self.variables:
                # Split violins: determine which side based on hue level
                hue_level = group_name[1] if isinstance(group_name, tuple) else None
                hue_levels = self.var_levels.get("hue", [])
                hue_idx = list(hue_levels).index(hue_level) if hue_level in hue_levels else 0

                if hue_idx == 0:  # Left side
                    fill_func(
                        support,
                        -self.offset + grid - density * self.dwidth,
                        -self.offset + grid,
                        facecolor=color,
                        **kws,
                    )
                else:  # Right side
                    fill_func(
                        support,
                        -self.offset + grid,
                        -self.offset + grid + density * self.dwidth,
                        facecolor=color,
                        **kws,
                    )
            else:
                # Half violin (left side only) - this is the core feature of half violins
                # The violin should extend from the center to the left, with offset applied
                fill_func(
                    support,
                    -self.offset + grid - density * self.dwidth,
                    -self.offset + grid,
                    facecolor=color,
                    **kws,
                )

            # Legend handling moved to plot() method using modern seaborn API

            # Draw the interior representation of the data
            if self.inner is not None:
                self._draw_violin_interior(ax, violin, center)

    def _get_center_position(self, group_name):
        """Get the numeric position on the categorical axis for a violin."""
        # Extract the primary category from group_name
        if "hue" in self.variables and isinstance(group_name, tuple) and len(group_name) >= 2:
            # With hue: group_name is like ('CategoryA', 'HueLevel1')
            primary_cat = group_name[0]

            # Get base position
            # Determine categorical variable based on orientation
            if "x" in self.variables and "y" in self.variables:
                if self.orient == "y":
                    # Horizontal: y is categorical
                    categorical_var = "y"
                else:
                    # Vertical: x is categorical
                    categorical_var = "x"
            else:
                categorical_var = "x" if self.orient == "v" else "y"
            base_pos = list(self.var_levels[categorical_var]).index(primary_cat)

            # For raincloud plots, we don't want hue to affect positioning
            # All violins should be positioned at the same offset relative to their category
            return base_pos
        else:
            # No hue: group_name is just the category
            if isinstance(group_name, tuple):
                primary_cat = group_name[0]
            else:
                primary_cat = group_name
            # Determine categorical variable based on orientation
            if "x" in self.variables and "y" in self.variables:
                if self.orient == "y":
                    # Horizontal: y is categorical
                    categorical_var = "y"
                else:
                    # Vertical: x is categorical
                    categorical_var = "x"
            else:
                categorical_var = "x" if self.orient == "v" else "y"
            return list(self.var_levels[categorical_var]).index(primary_cat)

    def _get_violin_color(self, group_name):
        """Get the color for a violin based on its group name."""
        if "hue" in self.variables and isinstance(group_name, tuple) and len(group_name) >= 2:
            # With hue: use hue mapping
            hue_level = group_name[1]
            return self._hue_map(hue_level)
        else:
            # No hue: determine the color to use
            # Priority: user_color > user_palette > _hue_map > default

            # 1. If user provided a specific color, use it (all categories same color)
            if self._user_color is not None:
                return self._user_color

            # 2. If user provided a palette, map each category to a color from the palette
            if self._user_palette is not None:
                if not hasattr(self, "_resolved_palette"):
                    self._resolved_palette = sns.color_palette(self._user_palette)

                # Get the category index to select the appropriate color
                if isinstance(group_name, tuple):
                    primary_cat = group_name[0]
                else:
                    primary_cat = group_name

                # Determine categorical variable based on orientation
                if "x" in self.variables and "y" in self.variables:
                    if self.orient == "y":
                        categorical_var = "y"
                    else:
                        categorical_var = "x"
                else:
                    categorical_var = "x" if self.orient == "v" else "y"

                cat_idx = list(self.var_levels[categorical_var]).index(primary_cat)
                # Cycle through palette if there are more categories than colors
                return self._resolved_palette[cat_idx % len(self._resolved_palette)]

            # 3. Try to get color from _hue_map
            if hasattr(self._hue_map, "lookup_table") and self._hue_map.lookup_table:
                return list(self._hue_map.lookup_table.values())[0]

            # 4. Fallback to default color
            return sns.color_palette()[0]

    def _draw_violin_interior(self, ax, violin, center):
        """Draw interior elements (box, quartiles, points, or sticks) for a violin."""
        observations = violin["observations"]
        support = violin["support"]
        density = violin["density"]

        # Handle split violins
        split_side = None
        if self.split and "hue" in self.variables:
            group_name = violin["group_name"]
            if isinstance(group_name, tuple):
                hue_level = group_name[1]
                hue_levels = self.var_levels.get("hue", [])
                hue_idx = list(hue_levels).index(hue_level) if hue_level in hue_levels else 0
                split_side = "left" if hue_idx == 0 else "right"

        # Draw interior elements based on inner style
        if self.inner.startswith("box"):
            self.draw_box_lines(ax, observations, support, density, center)
        elif self.inner.startswith("quart"):
            self.draw_quartiles(ax, observations, support, density, center, split_side)
        elif self.inner.startswith("stick"):
            self.draw_stick_lines(ax, observations, support, density, center, split_side)
        elif self.inner.startswith("point"):
            self.draw_points(ax, observations, center)

    def draw_single_observation(self, ax, at_group, at_quant, density):
        """Draw a line to mark a single observation."""
        d_width = density * self.dwidth
        if self.orient == "v":
            ax.plot(
                [at_group - d_width, at_group + d_width],
                [at_quant, at_quant],
                color=self.gray,
                linewidth=self.linewidth,
            )
        else:
            ax.plot(
                [at_quant, at_quant],
                [at_group - d_width, at_group + d_width],
                color=self.gray,
                linewidth=self.linewidth,
            )

    def draw_box_lines(self, ax, data, support, density, center):
        """Draw boxplot information at center of the density."""
        # Compute the boxplot statistics
        q25, q50, q75 = np.percentile(data, [25, 50, 75])
        whisker_lim = 1.5 * stats.iqr(data)
        h1 = np.min(data[data >= (q25 - whisker_lim)])
        h2 = np.max(data[data <= (q75 + whisker_lim)])

        # Draw a boxplot using lines and a point
        if self.orient == "v":
            ax.plot([center, center], [h1, h2], linewidth=self.linewidth, color=self.gray)
            ax.plot([center, center], [q25, q75], linewidth=self.linewidth * 3, color=self.gray)
            ax.scatter(
                center,
                q50,
                zorder=3,
                color="white",
                edgecolor=self.gray,
                s=np.square(self.linewidth * 2),
            )
        else:
            ax.plot([h1, h2], [center, center], linewidth=self.linewidth, color=self.gray)
            ax.plot([q25, q75], [center, center], linewidth=self.linewidth * 3, color=self.gray)
            ax.scatter(
                q50,
                center,
                zorder=3,
                color="white",
                edgecolor=self.gray,
                s=np.square(self.linewidth * 2),
            )

    def draw_quartiles(self, ax, data, support, density, center, split=None):
        """Draw the quartiles as lines at width of density."""
        q25, q50, q75 = np.percentile(data, [25, 50, 75])

        self.draw_to_density(
            ax,
            center,
            q25,
            support,
            density,
            split,
            linewidth=self.linewidth,
            dashes=[self.linewidth * 1.5] * 2,
        )
        self.draw_to_density(
            ax,
            center,
            q50,
            support,
            density,
            split,
            linewidth=self.linewidth,
            dashes=[self.linewidth * 3] * 2,
        )
        self.draw_to_density(
            ax,
            center,
            q75,
            support,
            density,
            split,
            linewidth=self.linewidth,
            dashes=[self.linewidth * 1.5] * 2,
        )

    def draw_points(self, ax, data, center):
        """Draw individual observations as points at middle of the violin."""
        kws = dict(s=np.square(self.linewidth * 2), color=self.gray, edgecolor=self.gray)

        grid = np.ones(len(data)) * center

        if self.orient == "v":
            ax.scatter(grid, data, **kws)
        else:
            ax.scatter(data, grid, **kws)

    def draw_stick_lines(self, ax, data, support, density, center, split=None):
        """Draw individual observations as sticks at width of density."""
        for val in data:
            self.draw_to_density(
                ax, center, val, support, density, split, linewidth=self.linewidth * 0.5
            )

    def draw_to_density(self, ax, center, val, support, density, split, **kws):
        """Draw a line orthogonal to the value axis at width of density."""
        idx = np.argmin(np.abs(support - val))
        width = self.dwidth * density[idx] * 0.99

        kws["color"] = self.gray

        if self.orient == "v":
            if split == "left":
                ax.plot([center - width, center], [val, val], **kws)
            elif split == "right":
                ax.plot([center, center + width], [val, val], **kws)
            else:
                ax.plot([center - width, center + width], [val, val], **kws)
        else:
            if split == "left":
                ax.plot([val, val], [center - width, center], **kws)
            elif split == "right":
                ax.plot([val, val], [center, center + width], **kws)
            else:
                ax.plot([val, val], [center - width, center + width], **kws)

    def plot(self, ax, kws):
        """Make the violin plot."""
        # Import the necessary helper from seaborn
        from seaborn.utils import _get_patch_legend_artist

        # Estimate densities for all violins first
        self.estimate_densities(self.bw, self.cut, self.scale, self.scale_hue, self.gridsize)

        self.draw_violins(ax, kws)

        # Set categorical tick labels explicitly
        # Modern seaborn doesn't automatically set these, so we need to do it manually
        cat_levels = self.var_levels[self.orient]
        n_cats = len(cat_levels)

        if self.orient == "x":
            ax.set_xticks(range(n_cats))
            ax.set_xticklabels(cat_levels)
            ax.set_xlim(-0.5, n_cats - 0.5)
            ax.xaxis.grid(False)
        else:  # orient == "y"
            ax.set_yticks(range(n_cats))
            ax.set_yticklabels(cat_levels)
            # For horizontal plots, limits are inverted (larger value first)
            ax.set_ylim(n_cats - 0.5, -0.5)
            ax.yaxis.grid(False)

        # Note: We intentionally don't set axis labels here.
        # In seaborn 0.11, annotate_axes() set labels, but in 0.13 the behavior changed.
        # For standalone plots, users can set labels manually.
        # For FacetGrid plots, FacetGrid handles labels itself.
        # This matches the original behavior where FacetGrid plots had no labels from
        # the individual plotting functions.

        # Configure the legend correctly using the modern Seaborn API
        # Only configure legend if hue is present and not redundant
        if "hue" in self.variables and not self._redundant_hue:
            legend_artist = _get_patch_legend_artist(fill=True)
            common_kws = {"facecolor": "C0", "edgecolor": self.gray, "linewidth": self.linewidth}
            self._configure_legend(ax, legend_artist, common_kws)


def stripplot(
    x: DataInput = None,
    y: DataInput = None,
    hue: DataInput = None,
    data: Optional[pd.DataFrame] = None,
    order: Optional[list[str]] = None,
    hue_order: Optional[list[str]] = None,
    jitter: bool = True,
    dodge: bool = False,
    orient: Optional[str] = None,
    color: Optional[str] = None,
    palette: Optional[Union[str, list, dict]] = None,
    move: float = 0,
    size: float = 5,
    edgecolor: str = "gray",
    linewidth: float = 0,
    ax: Optional[matplotlib.axes.Axes] = None,
    width: float = 0.8,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    """
    A wrapper around seaborn's stripplot that adds a `move` parameter
    and preserves specific style defaults.

    Parameters
    ----------
    move : float, default 0
        Shift the strip plot along the categorical axis to create offset positioning.
        Positive values move the points in the positive direction of the categorical axis.

    Other parameters are passed directly to seaborn.stripplot().
    See seaborn.stripplot documentation for full parameter details.
    """
    # 1. Handle legacy `split` argument if necessary
    if "split" in kwargs:
        dodge = kwargs.pop("split")
        warnings.warn("The `split` parameter has been renamed to `dodge`.", UserWarning)

    # 2. Get the current axes if one isn't provided
    if ax is None:
        ax = plt.gca()

    # 3. Future-proof for seaborn 0.14: auto-set hue when palette is provided
    # This avoids the deprecation warning and ensures colors work correctly
    legend = kwargs.pop("legend", None)  # Extract legend if passed in kwargs
    if palette is not None and hue is None and data is not None and x is not None and y is not None:
        # Only auto-set hue when x and y are column names (strings), not raw data arrays
        # This ensures we work correctly with both DataFrame and array inputs
        if isinstance(x, str) and isinstance(y, str):
            # Set hue to the categorical variable to enable per-category coloring
            # Determine which variable is categorical based on orient
            if orient in ["h", "y"]:
                hue = y
            elif orient in ["v", "x"]:
                hue = x
            else:
                # If orient not specified, infer from data types
                # Default to x being categorical (vertical plot)
                hue = x

            # Suppress legend since hue is just being used for coloring, not semantic meaning
            if legend is None:
                legend = False

    # 4. Apply old version's smart parameter processing
    if linewidth is None:
        linewidth = size / 10
    if edgecolor == "gray":
        # Convert "gray" string to actual gray color for backward compatibility
        edgecolor = _LEGACY_GRAY

    # 5. Call the official seaborn stripplot function.
    #    We pass all standard arguments directly to it.
    pre_strip_collections = len(ax.collections)

    # Build kwargs for seaborn, including legend if it was set
    seaborn_kwargs = kwargs.copy()
    if legend is not None:
        seaborn_kwargs["legend"] = legend

    sns.stripplot(
        x=x,
        y=y,
        hue=hue,
        data=data,
        order=order,
        hue_order=hue_order,
        jitter=jitter,
        dodge=dodge,
        orient=orient,
        color=color,
        palette=palette,
        size=size,
        edgecolor=edgecolor,
        linewidth=linewidth,
        ax=ax,
        **seaborn_kwargs,
    )

    # 4. Fix dodge offset to match boxplot width when dodge=True
    # Seaborn's stripplot uses width=0.8 internally for dodge calculations,
    # but we want dodge to match the boxplot width for proper alignment
    new_collections = ax.collections[pre_strip_collections:]

    if dodge and width != 0.8:
        # Seaborn uses width=0.8 internally for dodge calculations
        # We need to rescale positions to match the user-specified width
        # Position = center + dodge_offset + jitter
        # Seaborn calculates dodge based on 0.8, but we want it based on 'width'

        # However, we DON'T want to scale jitter - we want full jitter range
        # The issue is seaborn reduces jitter when dodge=True: jitter /= n_hue_levels
        # So we need to: scale dodge down, but keep jitter at original magnitude

        dodge_scale = width / 0.8

        for points_collection in new_collections:
            if not hasattr(points_collection, "get_offsets"):
                continue

            offsets = points_collection.get_offsets()
            if offsets is None or len(offsets) == 0:
                continue

            # Matplotlib can return a masked array; operate on a numpy copy.
            offsets = np.asarray(offsets)

            # Determine which axis is categorical based on orient
            if orient in ["h", "y"]:
                cat_axis = 1  # y-axis
            else:
                cat_axis = 0  # x-axis

            # Get the categorical positions
            cat_positions = offsets[:, cat_axis]
            cat_centers = np.round(cat_positions)

            # Total offset from center includes both dodge and jitter
            total_offsets = cat_positions - cat_centers

            # We need to separate dodge from jitter, but they're mixed together
            # Dodge offsets are deterministic per hue level, jitter is random per point
            # Strategy: assume the median offset per collection is the dodge offset
            median_offset = np.median(total_offsets)
            jitter_components = total_offsets - median_offset

            # Scale only the dodge component
            scaled_dodge = median_offset * dodge_scale

            # Recombine: scaled dodge + original jitter
            new_positions = cat_centers + scaled_dodge + jitter_components

            offsets[:, cat_axis] = new_positions
            points_collection.set_offsets(offsets)

    # 5. Apply the custom `move` functionality if needed
    if move != 0:
        # Iterate over every new PathCollection emitted by seaborn.stripplot.
        for points_collection in new_collections:
            if not hasattr(points_collection, "get_offsets"):
                continue

            offsets = points_collection.get_offsets()
            if offsets is None or len(offsets) == 0:
                continue

            # Matplotlib can return a masked array; operate on a numpy copy.
            offsets = np.asarray(offsets)

            # Check orientation to decide which axis to shift
            if orient in ["h", "y"]:
                # Horizontal plot: move the y-positions
                offsets[:, 1] += move
            else:
                # Vertical plot: move the x-positions
                offsets[:, 0] += move

            points_collection.set_offsets(offsets)

    # 5. Return the axes object
    return ax


def half_violinplot(
    x: DataInput = None,
    y: DataInput = None,
    hue: DataInput = None,
    data: Optional[pd.DataFrame] = None,
    order: Optional[list[str]] = None,
    hue_order: Optional[list[str]] = None,
    bw: Union[str, float] = "scott",
    cut: float = 2,
    scale: str = "area",
    scale_hue: bool = True,
    gridsize: int = 100,
    width: float = 0.8,
    inner: Optional[str] = "box",
    split: bool = False,
    dodge: bool = True,
    orient: Optional[str] = None,
    linewidth: Optional[float] = None,
    color: Optional[str] = None,
    palette: Optional[Union[str, list, dict]] = None,
    saturation: float = 0.75,
    ax: Optional[matplotlib.axes.Axes] = None,
    offset: float = 0.15,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    plotter = _Half_ViolinPlotter(
        x,
        y,
        hue,
        data,
        order,
        hue_order,
        bw,
        cut,
        scale,
        scale_hue,
        gridsize,
        width,
        inner,
        split,
        dodge,
        orient,
        linewidth,
        color,
        palette,
        saturation,
        offset,
    )

    if ax is None:
        ax = plt.gca()

    plotter.plot(ax, kwargs)
    return ax


def RainCloud(
    x: DataInput = None,
    y: DataInput = None,
    hue: DataInput = None,
    data: Optional[pd.DataFrame] = None,
    order: Optional[list[str]] = None,
    hue_order: Optional[list[str]] = None,
    orient: str = "v",
    width_viol: float = 0.7,
    width_box: float = 0.15,
    palette: Optional[Union[str, list, dict]] = None,
    bw: Union[str, float] = 0.2,
    linewidth: float = 1,
    cut: float = 0.0,
    scale: str = "area",
    jitter: bool = True,
    move: float = 0.0,
    offset: Optional[float] = None,
    point_size: float = 3,
    ax: Optional[matplotlib.axes.Axes] = None,
    pointplot: bool = False,
    alpha: Optional[float] = None,
    dodge: bool = False,
    linecolor: str = "red",
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    """Draw a Raincloud plot of measure `y` of different categories `x`.

    Here `x` and `y` are different columns of the pandas dataframe `data`.

    A raincloud is made of:
        1) "Cloud", kernel desity estimate, the half of a violinplot.
        2) "Rain", a stripplot below the cloud
        3) "Umberella", a boxplot
        4) "Thunder", a pointplot connecting the mean of different categories
           (if `pointplot` is `True`)

    Main inputs:
        x           categorical data. Iterable, np.array, or dataframe column
                    name if 'data' is specified
        y           measure data. Iterable, np.array, or dataframe column name
                    if 'data' is specified
        hue         a second categorical data. Use it to obtain different
                    clouds and rainpoints
        data        input pandas dataframe
        order       list, order of the categorical data
        hue_order   list, order of the hue
        orient      string, vertical if "v" (default), horizontal if "h"
        width_viol  float, width of the cloud
        width_box   float, width of the boxplot
        move        float, adjusts rain position to the x-axis (default 0.)
        offset      float, adjusts cloud position to the x-axis

    kwargs can be passed to the [cloud (default), boxplot, rain/stripplot,
    pointplot] by preponing [cloud_, box_, rain_ point_] to the argument name.
    """

    # Save original variable names for axis labels before swapping
    orig_x, orig_y = x, y

    if orient == "h":  # swap x and y
        x, y = y, x
    if ax is None:
        ax = plt.gca()

    # Handle seaborn 0.14+ deprecation: palette without hue
    # When palette is provided but hue is None, auto-assign hue to categorical variable
    _internal_hue = hue
    _suppress_legend = False
    if palette is not None and hue is None:
        # Assign hue to the categorical variable (x for vertical, y for horizontal)
        _internal_hue = x if orient == "v" else y
        _suppress_legend = True  # Legend would be redundant with axis labels

    if offset is None:
        offset = max(width_box / 1.8, 0.15) + 0.05
    n_plots = 3
    split = False
    boxcolor = "black"
    boxprops = {"facecolor": "none", "zorder": 10}

    # Determine if hue represents actual subgroups (different from categorical variable)
    categorical_var = y if orient == "h" else x
    categorical_var_name = categorical_var if isinstance(categorical_var, str) else None
    hue_name = _internal_hue if isinstance(_internal_hue, str) else None
    has_subgroups = _internal_hue is not None and hue_name != categorical_var_name

    if _internal_hue is not None:
        boxcolor = palette
        if has_subgroups:
            boxprops = {"zorder": 10}
        else:
            boxprops = {"facecolor": "none", "zorder": 10}

    kwcloud = dict()
    kwbox = dict(saturation=1, whiskerprops={"linewidth": 2, "zorder": 10})
    kwrain = dict(zorder=0, edgecolor="white")
    kwpoint = dict(capsize=0.0, errwidth=0.0, zorder=20)
    for key, value in kwargs.items():
        if "cloud_" in key:
            kwcloud[key.replace("cloud_", "")] = value
        elif "box_" in key:
            kwbox[key.replace("box_", "")] = value
        elif "rain_" in key:
            kwrain[key.replace("rain_", "")] = value
        elif "point_" in key:
            kwpoint[key.replace("point_", "")] = value
        else:
            kwcloud[key] = value

    # Draw cloud/half-violin
    half_violinplot(
        x=x,
        y=y,
        hue=_internal_hue,
        data=data,
        order=order,
        hue_order=hue_order,
        orient=orient,
        width=width_viol,
        inner=None,
        palette=palette,
        bw=bw,
        linewidth=linewidth,
        cut=cut,
        scale=scale,
        split=split,
        offset=offset,
        ax=ax,
        **kwcloud,
    )

    # Draw umberella/boxplot
    sns.boxplot(
        x=x,
        y=y,
        hue=_internal_hue,
        data=data,
        orient=orient,
        width=width_box,
        order=order,
        hue_order=hue_order,
        color=boxcolor,
        showcaps=True,
        boxprops=boxprops,
        palette=palette,
        dodge=dodge,
        legend=False if _suppress_legend else "auto",
        ax=ax,
        **kwbox,
    )

    # Set alpha for violin and boxplot elements
    # This affects PolyCollections (violin), PathPatches (boxplot boxes), and Lines (boxplot whiskers)
    if alpha is not None:
        # Apply to all collections (violin patches)
        for collection in ax.collections:
            collection.set_alpha(alpha)
        # Apply to all patches (boxplot boxes)
        for patch in ax.patches:
            patch.set_alpha(alpha)
        # Apply to all artists (if any)
        for artist in ax.artists:
            artist.set_alpha(alpha)
        # Apply to lines (boxplot whiskers, caps, medians)
        for line in ax.lines:
            line.set_alpha(alpha)

    # Draw rain/stripplot
    ax = stripplot(
        x=x,
        y=y,
        hue=hue,
        data=data,
        orient=orient,
        order=order,
        hue_order=hue_order,
        palette=palette,
        move=move,
        size=point_size,
        jitter=jitter,
        dodge=dodge,
        width=width_box,
        ax=ax,
        **kwrain,
    )

    # Add pointplot
    if pointplot:
        n_plots = 4
        # When hue is present and represents subgroups, show separate lines per hue level
        # Otherwise show a single line
        if hue is not None and hue_name != categorical_var_name:
            sns.pointplot(
                x=x,
                y=y,
                hue=hue,
                data=data,
                palette=palette,
                orient=orient,
                order=order,
                hue_order=hue_order,
                linestyles="-",
                ax=ax,
                **kwpoint,
            )
        else:
            sns.pointplot(
                x=x,
                y=y,
                data=data,
                color=linecolor,
                orient=orient,
                order=order,
                linestyles="-",
                ax=ax,
                **kwpoint,
            )

    # Prune the legend, add legend title
    # Only show legend if hue is a different variable than the categorical axis
    # (otherwise the legend is redundant with the axis labels)
    # Use the names we calculated earlier to avoid Series comparison issues
    if hue is not None and hue_name != categorical_var_name:
        handles, labels = ax.get_legend_handles_labels()

        # Each plot component (violin, box, strip) adds its own legend entries when `hue`
        # is used. We only want to show one set of entries for clarity.
        num_hue_levels = len(labels) // n_plots
        _ = plt.legend(
            handles[:num_hue_levels],
            labels[:num_hue_levels],
            bbox_to_anchor=(1.05, 1),
            loc=2,
            borderaxespad=0.0,
            title=str(hue),
        )  # , title_fontsize = 25)
    else:
        # Remove any legend that was created by the plotting functions
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    # Set axis labels based on ORIGINAL variable names
    # When orient='h', the user expects: categorical var on y-axis, numeric var on x-axis
    # When orient='v', the user expects: categorical var on x-axis, numeric var on y-axis
    # Note: In FacetGrid context, these will be overridden by FacetGrid._finalize_grid
    # Users should manually clear them in notebook if using FacetGrid with kwargs
    if orient == "h":
        # Horizontal: categorical (orig_x) on y-axis, numeric (orig_y) on x-axis
        if isinstance(orig_y, str):
            ax.set_xlabel(orig_y)
        if isinstance(orig_x, str):
            ax.set_ylabel(orig_x)
    else:
        # Vertical: categorical (orig_x) on x-axis, numeric (orig_y) on y-axis
        if isinstance(orig_x, str):
            ax.set_xlabel(orig_x)
        if isinstance(orig_y, str):
            ax.set_ylabel(orig_y)

    # Adjust the ylim to fit (if needed)
    if orient == "h":
        ylim = list(ax.get_ylim())
        ylim[-1] -= (width_box + width_viol) / 4.0
        _ = ax.set_ylim(ylim)
    elif orient == "v":
        xlim = list(ax.get_xlim())
        xlim[-1] -= (width_box + width_viol) / 4.0
        _ = ax.set_xlim(xlim)

    return ax
