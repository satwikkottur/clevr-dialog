"""Utilities for CLEVR-Dialog dataset generation.

Author: Satwik Kottur
"""

import copy


def pretty_print_templates(templates, verbosity=1):
  """Pretty prints templates.

  Args:
    templates: Templates to print
    verbosity: 1 to print name and type of the templates
  """

  # Verbosity 1: Name and type.
  print('-'*70)
  for ii in templates:
    print('[Name: %s] [Type: %s]' % (ii['name'], ii['type']))
  print('-'*70)
  print('Total of %s templates..' % len(templates))
  print('-'*70)


def pretty_print_scene_objects(scene):
  """Pretty prints scene objects.

  Args:
    scene: Scene graph containing list of objects
  """

  for index, ii in enumerate(scene['objects']):
    print_args = (index, ii['shape'], ii['color'], ii['size'], ii['material'])
    print('\t%d : %s-%s-%s-%s' % print_args)


def pretty_print_dialogs(dialogs):
  """Pretty prints generated dialogs.

  Args:
    dialogs: Generated dialogs to print
  """

  for scene_id, dialog_datum in enumerate(dialogs):
    for dialog in dialog_datum['dialogs']:
      print(dialog['caption'])
      for round_id, ii in enumerate(dialog['dialog']):
        coref_id = dialog['graph']['history'][round_id+1]['dependence']
        in_tuple = (round_id, ii['question'], str(ii['answer']),
                ii['template'], str(coref_id))
        print('\t[Q-%d: %s] [A: %s] [%s] [%s]' % in_tuple)


def merge_update_scene_graph(orig_graph, graph_item):
  """Merges two scene graphs into one.

  Args:
    orig_graph: Original scene graph
    graph_item: New graph item to add to the scene graph

  Returns:
    graph: Deep copy of the original scene graph after merging
  """

  graph = copy.deepcopy(orig_graph)
  # Local alias.
  objects = graph['objects']

  # If not mergeable, return the same scene graph.
  if not graph_item['mergeable']:
    return graph

  # 1. Go through each new object
  # 2. Find its batch in objects
  #   a. If found, assert for a clash of attributes, update
  #   b. If novel, just add the object as is
  for new_obj in graph_item['objects']:
    match_found = False
    obj = objects.get(new_obj['id'], None)

    if obj:
      # Assert for existing entries.
      for attr in new_obj:
        try:
          assert new_obj[attr] == obj.get(attr, new_obj[attr]),\
            'Some of the attributes do not match!'
        except: pdb.set_trace()

      # Add additional keys.
      objects[new_obj['id']].update(new_obj)
    else:
      # Add the new object.
      objects[new_obj['id']] = new_obj

  # if a relation, update it
  if 'relation' in graph_item:
    rel = graph_item['relation']
    ## update it with object 2 id
    id1 = graph_item['objects'][0]['id']
    id2 = graph_item['objects'][1]['id']
    rel_objs = graph['relationships'][rel][id1]
    rel_objs.append(id2)
    graph['relationships'][rel][id1] = rel_objs

  # update objects in graph
  graph['objects'] = objects
  return graph


def add_object_ids(scenes):
  """Adds object ids field for input scenes.

  Args:
    scenes: List of CLEVR scene graphs

  Returns:
    scenes: Adds object_id field for the objects in the scene graph inplace
  """

  for scene_id, scene in enumerate(scenes['scenes']):
    for obj_id, _ in enumerate(scene['objects']):
      scenes['scenes'][scene_id]['objects'][obj_id]['id'] = obj_id
  return scenes


def clean_object_attributes(scenes):
  """Cleans attributes for objects, keeping only attributes and id.

  Args:
    scenes: Scene graph to clean

  Returns:
    scenes: Cleaned up scene graphs inplace
  """

  keys = ['shape', 'size', 'material', 'color', 'id']
  for scene_id, scene in enumerate(scenes['scenes']):
    for obj_id, obj in enumerate(scene['objects']):
      new_obj = {key: obj[key] for key in keys}
      scenes['scenes'][scene_id]['objects'][obj_id] = new_obj
  return scenes


def pretty_print_corefs(dialog, coref_groups):
  """Prints coreferences for a dialog, higlighting different groups in colors.

  Args:
    dialog: Generated dialogs to print
    coref_groups: Coreference groups for dialogs
  """

  colorama.init()
  # Mapping of group_id -> color_ids for (foreground, background)
  color_map = {}
  groups = coref_groups.get(0, [])
  colored, color_map = pretty_print_coref_sentence(dialog['caption'], groups,
                                                   color_map)
  print('\n\nC: %s' % colored)
  for round_id, round_datum in enumerate(dialog['dialog']):
    question = round_datum['question']
    groups = coref_groups.get(round_id + 1, [])
    colored, color_map = pretty_print_coref_sentence(question, groups,
                                                     color_map)
    print('%d: %s' % (round_id, colored))


def pretty_print_coref_sentence(sentence, groups, color_map):
  """Prints a sentence containing difference coreference groups.

  Args:
    sentence: Text sentence
    groups: List of coreference groups with spans
    color_map: List of groups and associated color maps

  Returns:
    sentence: Text sentence with colors inserted
    color_map: Updated, if new groups in the current sentence
  """

  fore_colors = ['RED', 'GREEN', 'YELLOW', 'BLUE', 'MAGENTA']
  back_colors = ['BLACK', 'YELLOW', 'CYAN']
  insertions = []
  for group in groups:
    group_id = group['group_id']
    if group_id in color_map:
      forecolor_id, backcolor_id = color_map[group_id]
    else:
      num_groups = len(color_map)
      forecolor_id = num_groups % len(fore_colors)
      backcolor_id = num_groups // len(fore_colors)
      color_map[group_id] = (forecolor_id, backcolor_id)

    forecolor = fore_colors[forecolor_id]
    backcolor = back_colors[backcolor_id]
    insertions.append((group['span'][0], getattr(colorama.Fore, forecolor)))
    insertions.append((group['span'][0], getattr(colorama.Back, backcolor)))
    insertions.append((group['span'][1],
                       getattr(colorama.Style, 'RESET_ALL')))

  # Perform insertions.
  sentence = insert_into_sentence(sentence, insertions)
  return sentence, color_map


def insert_into_sentence(sentence, insertions):
  """Sorts and performs insertions from right.

  Args:
    sentence: Sentence to perform insertions into
    insertions: List of insertions, format: (position, text_insert)

  Returns:
    sentence: Inplace inserted sentence
  """

  insertions = sorted(insertions, key=lambda x: x[0], reverse=True)
  for position, text in insertions:
    sentence = sentence[:position] + text + sentence[position:]
  return sentence
