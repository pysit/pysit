"""cloud_sptheme.ext.overrides -- override various minor sphinx behaviors

* override_sidebar_logo -- glob map (ala html_sidebars), use to change sidebar logo per-page
* override_localtoc_title -- change local toc title (requires support in localtoc.html template)

.. todo::
    document this extension
"""
import os.path
import re
import logging; log = logging.getLogger(__name__)
from sphinx.util.matching import patmatch

def _rank_pattern(pattern):
    """return sorting key for prioritizing which glob pattern should match"""
    # TODO: add more ways to distinguish patterns
    return not any(char in pattern for char in '*?[')

def bestmatch(patmap, source, default=None, param="source"):
    """return best match given a dictionary mapping glob pattersn -> values"""
    best = None
    best_rank = None
    for pattern in patmap:
        if not patmatch(pattern, source):
            continue
        cur_rank = _rank_pattern(pattern)
        if best is None or cur_rank < best_rank:
            best = pattern
            best_rank = cur_rank
        elif cur_rank == best_rank:
            raise KeyError("%s %r matches too many patterns: %r and %r" %
                           (param, source, best, pattern))
    if best is None:
        return default
    else:
        return patmap[best]

def override_sidebar_logo(app, pagename, templatename, ctx, event_arg):
    """helper to override sidebar logo per-page"""
    patmap = getattr(app.config, "override_sidebar_logo", None)
    if not patmap:
        return
    logo = bestmatch(patmap, pagename, ctx.get("logo"), param="pagename")
    if logo is None:
        ctx.pop("logo", None)
    else:
        ctx['logo'] = logo

def override_localtoc_title(app, pagename, templatename, ctx, event_arg):
    title = getattr(app.config, "override_localtoc_title", None)
    if title:
        ctx['localtoc_title'] = title

##def override_template_context(app, pagename, templatename, ctx, event_arg):
##    patmap = getattr(app.config, "override_template_context")
##    if not patmap:
##        return
##    values = bestmatch(patmap, pagename, None, param="pagename")
##    if values:
##        ctx.update(values)

def setup(app):
    app.add_config_value('override_sidebar_logo', None, 'env')
    app.add_config_value('override_localtoc_title', None, 'env')
    ##app.add_config_value('override_template_context', None, 'env')
    app.connect('html-page-context', override_sidebar_logo)
    app.connect('html-page-context', override_localtoc_title)
    ##app.connect('html-page-context', override_template_context)
