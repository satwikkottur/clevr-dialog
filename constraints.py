"""Supporting script to check constraints for caption and question generation.

Author: Satwik Kottur
"""

import copy
import json
import random
import pdb
import numpy as np


with open('metainfo.json', 'r') as file_id:
  meta_info = json.load(file_id)

attributes = meta_info['attributes']
values = meta_info['values']
relations = meta_info['relations']
# global configs
probs = [0.70, 0.30]

# some quick methods
apply_immediate = lambda hist: (len(hist['objects']) == 1 and hist['mergeable']
                                and 'exist' not in hist['template'])
apply_group = lambda hist: (len(hist['objects']) >= 2 and hist['mergeable']
                            and 'count' not in prev_group)


def caption(scene, templates):
  """Constraints for caption generation.

  Args:
    scene:
    template:

  Returns:
  """

  caption_hypotheses = {}

  # Sweep through all templates to extract 'interesting' captions.
  n_objs = len(scene['objects'])
  rels = scene['relationships']

  # Caption Type 1: Extreme locations.
  ext_loc_templates = [ii for ii in templates if ii['type'] == 'extreme-loc']
  # number of objects in the scene
  filter_objs = copy.deepcopy(scene['objects'])
  attr_counts = get_attribute_counts_for_objects(scene, filter_objs)
  hypotheses = []
  for template in ext_loc_templates:
    # absolute location based constraint
    constraint = template['constraints'][0]
    extreme_type = constraint['args'][0]

    # check if there is an object that is at the center of the image
    # roughly in the middle along front-back and right-left dim
    if extreme_type == 'center':
      for ii, obj in enumerate(filter_objs):
        matches = np.sum([len(rels[kk][ii]) <= n_objs / 2
                          for kk in ['front', 'behind', 'right', 'left']])
        if matches == 4:
          hypotheses.append((extreme_type, copy.deepcopy(obj)))
    else:
      for ii, obj in enumerate(filter_objs):
        if len(rels[extreme_type][ii]) == 0:
          hypotheses.append((extreme_type, copy.deepcopy(obj)))

  # sample one at random, and create the graph item
  # Filter hypothesis which are ambiguous otherwise.
  for index, (_, hypothesis) in enumerate(hypotheses):
    uniq_attr = [attr for attr in attributes
                 if attr_counts[(attr, hypothesis[attr])] == 1]

    for attr in uniq_attr:
      del hypotheses[index][1][attr]

  hypotheses = [ii for ii in hypotheses if len(ii[1]) > 1]
  caption_hypotheses['extreme-loc'] = hypotheses

  # Caption Type 2: Unique object and attribute.
  filter_objs = copy.deepcopy(scene['objects'])
  # each hypothesis is (object, attribute) pair
  hypotheses = []
  for ii, obj in enumerate(filter_objs):
    # get unique set of attributes
    uniq_attrs = [ii for ii in attributes if attr_counts[(ii, obj[ii])] == 1]
    # for each, add it to hypothesis
    for attr in uniq_attrs:
      hypotheses.append((obj, attr))
  caption_hypotheses['unique-obj'] = hypotheses

  # Caption Type 3: Unique attribute count based caption.
  # count unique object based constraint
  # Each hypothesis is object collection.
  caption_hypotheses['count-attr'] = [(attr_val, count)
                                      for attr_val, count in attr_counts.items()
                                      if count > 1]


  # Caption Type 4: Relation between two objects.
  # Out of the two, one has a unique attribute.
  # find a pair of objects sharing a relation, unique
  filter_objs = copy.deepcopy(scene['objects'])
  n_objs = len(filter_objs)

  # get a dict of unique attributes for each object
  uniq_attr = [[] for ii in range(n_objs)]
  non_uniq_attr = [[] for ii in range(n_objs)]
  for ind, obj in enumerate(filter_objs):
    uniq_attr[ind] = [attr for attr in attributes
                      if attr_counts[(attr, obj[attr])] == 1]
    non_uniq_attr[ind] = [attr for attr in attributes
                          if attr_counts[(attr, obj[attr])] > 1]
  uniqueness = [len(ii) > 0 for ii in uniq_attr]

  # Hypothesis is a uniq object and non-unique obj2 sharing relation R
  # global ordering for uniqueness
  hypotheses = []
  for rel, order in scene['relationships'].items():
    num_rel = [(ii, len(order[ii])) for ii in range(n_objs)]
    num_rel = sorted(num_rel, key=lambda x:x[1], reverse=True)
    # take only the ids
    num_rel = [ii[0] for ii in num_rel]

    for index, obj_id in enumerate(num_rel[:-1]):
      next_obj_id = num_rel[index + 1]
      # if unique, check if the next one has non-unique attributes
      if uniqueness[obj_id]:
        if len(non_uniq_attr[next_obj_id]) > 0:
          obj1 = (obj_id, random.choice(uniq_attr[obj_id]))
          obj2 = (next_obj_id, random.choice(non_uniq_attr[next_obj_id]))
          hypotheses.append((obj1, rel, obj2))
      # if not unique, check if the next one has unique attributes
      else:
        if len(uniq_attr[next_obj_id]) > 0:
          obj1 = (obj_id, random.choice(non_uniq_attr[obj_id]))
          obj2 = (next_obj_id, random.choice(uniq_attr[next_obj_id]))
          hypotheses.append((obj1, rel, obj2))
  caption_hypotheses['obj-relation'] = hypotheses
  return sample_from_hypotheses(caption_hypotheses, scene, templates)


def sample_from_hypotheses(caption_hypotheses, scene, cap_templates):
  obj_groups = []

  # Caption Type 1: Extreme location.
  hypotheses = caption_hypotheses['extreme-loc']
  if len(hypotheses) > 0:
    # extreme location hypotheses
    extreme_type, focus_obj = random.choice(hypotheses)
    # sample optional attributes
    obj_attrs = [attr for attr in attributes if attr in focus_obj]
    focus_attr = random.choice(obj_attrs)
    optional_attrs = [ii for ii in obj_attrs if ii != focus_attr]
    sampled_attrs = sample_optional_tags(optional_attrs, probs)

    # add additional attributes
    req_attrs = sampled_attrs + [focus_attr]
    filter_obj = {attr: val for attr, val in focus_obj.items()
                  if attr in req_attrs}
    filter_obj['required'] = req_attrs
    filter_obj['optional'] = req_attrs
    filter_obj['id'] = focus_obj['id']
    obj_group = {'required': req_attrs, 'optional': [], 'group_id': 0,
                 'objects': [filter_obj]}

    # also create a clean graph object
    graph_item = copy.deepcopy(obj_group)
    graph_item = clean_graph_item(graph_item)
    graph_item['mergeable'] = True
    graph_item['objects'][0]['%s_count' % extreme_type] = 0
    graph_item['objects'][0]['%s_exist' % extreme_type] = False
    graph_item['template'] = 'extreme-%s' % extreme_type
    obj_group['graph'] = graph_item
    obj_groups.append([obj_group])

  # Caption Type 2: Unique object.
  hypotheses = caption_hypotheses['unique-obj']
  if len(hypotheses) > 0:
    # sample one at random, and create the graph item
    focus_obj, focus_attr = random.choice(hypotheses)
    # sample optional attributes
    optional_attrs = [ii for ii in attributes if ii != focus_attr]
    sampled_attrs = sample_optional_tags(optional_attrs, probs)

    # add additional attributes
    req_attrs = sampled_attrs + [focus_attr]
    filter_obj = {attr:val for attr, val in focus_obj.items()\
                      if attr in req_attrs}
    filter_obj['required'] = req_attrs
    filter_obj['optional'] = req_attrs
    filter_obj['id'] = focus_obj['id']
    obj_group = {'required': req_attrs, 'optional': [], 'group_id': 0,
                 'objects': [filter_obj]}

    # also create a clean graph object
    graph_item = copy.deepcopy(obj_group)
    graph_item = clean_graph_item(graph_item)
    graph_item['mergeable'] = True
    graph_item['objects'][0]['unique'] = True
    graph_item['template'] = 'unique-obj'
    obj_group['graph'] = graph_item
    obj_groups.append([obj_group])

  # Caption Type 3: Unique attribute count based caption.
  hypotheses = caption_hypotheses['count-attr']
  if len(hypotheses) > 0:
    # Randomly sample one hypothesis and one template.
    (attr, value), count = random.choice(hypotheses)
    # Segregate counting templates.
    count_templates = [ii for ii in cap_templates if 'count' in ii['type']]
    template = random.choice(count_templates)
    obj_group = {'group_id': 0, 'count': count, attr: value,
                 'optional': [], 'required': [], 'objects': []}

    # get a list of objects which are part of this collection
    for ii, obj in enumerate(scene['objects']):
      if obj[attr] == value:
        new_obj = {'id': obj['id'], attr: value}
        new_obj['required'] = [attr]
        new_obj['optional'] = []
        obj_group['objects'].append(new_obj)

    if 'no' in template['label']:
      # Count is not mentioned.
      del obj_group['count']
      graph_item = copy.deepcopy(obj_group)
      graph_item['mergeable'] = False
    else:
      # Count is mentioned.
      for index, ii in enumerate(obj_group['objects']):
        obj_group['objects'][index]['required'].append('count')
      graph_item = copy.deepcopy(obj_group)
      graph_item['mergeable'] = True

    # clean up graph item
    graph_item['template'] = template['label']
    graph_item = clean_graph_item(graph_item)
    obj_group['graph'] = graph_item
    obj_group['use_plural'] = True
    obj_groups.append([obj_group])

  # Caption Type 4: Relation between two objects (one of them is unique).
  hypotheses = caption_hypotheses['obj-relation']
  if len(hypotheses) > 0:
    (obj_id1, attr1), rel, (obj_id2, attr2) = random.choice(hypotheses)
    obj_group = {'group_id': 0, 'relation': rel}

    # create object dictionaries
    obj1 = {'optional': [], 'required': [attr1], 'id': obj_id1,
            attr1: scene['objects'][obj_id1][attr1]}
    obj2 = {'optional': [], 'required': [attr2], 'id': obj_id2,
            attr2: scene['objects'][obj_id2][attr2]}
    obj_group['objects'] = [obj2, obj1]

    # also create a clean graph object
    graph_item = copy.deepcopy(obj_group)
    graph_item = clean_graph_item(graph_item)
    graph_item['mergeable'] = True
    graph_item['template'] = 'obj-relation'
    obj_group['graph'] = graph_item
    obj_groups.append([obj_group])
  return obj_groups


def question(scene, dialog, template):
  """
  Inputs:
    scene: The scene so far
    template: which template to use

  Output:
    list of object groups
  """

  ques_round = len(dialog['graph']['history']) - 1
  graph = dialog['graph']

  # check for constraints and answer question
  if 'group' in template['label']:
    groups = []
    # pick a group hypothesis
    for ii in graph['history']:
      if 'count' in ii or len(ii['objects']) == 0:
        groups.append(ii)

  #----------------------------------------------------------------
  if template['label'] == 'count-all':
    # preliminary checks:
    # (A) count-all cannot follow count-all, count-other
    for prev_history in graph['history'][1:]:
      if prev_history['template'] in ['count-all', 'count-other']:
        return []

    # create object group
    obj_group = []
    new_obj = {'required': [], 'optional': []}
    for obj_id, ii in enumerate(scene['objects']):
      obj_copy = copy.deepcopy(new_obj)
      obj_copy['id'] = ii['id']
      obj_group.append(obj_copy)

    # create graph item
    graph_item = {'round': ques_round + 1,
                  'objects': copy.deepcopy(obj_group),
                  'template': template['label'],
                  'mergeable': True, 'count': len(obj_group)}
    # clean graph item
    graph_item = clean_graph_item(graph_item)
    # no constraints, count the number of objects in true scene
    return [{'answer': len(obj_group), 'group_id': ques_round + 1,
             'objects': [], 'graph': graph_item}]
  #----------------------------------------------------------------

  elif template['label'] == 'count-other' \
        or template['label'] == 'exist-other':
    # preliminary checks:
    # (A) exist-other cannot follow exist-other, count-all, count-other
    # (B) count-other cannot follow count-all, count-other
    for prev_history in graph['history'][1:]:
      if prev_history['template'] in ['count-all', 'count-other']:
        return []

      if prev_history['template'] == 'exist-other' and \
            template['label'] == 'exist-other':
        return []

    # get a list of all objects we know
    known_ids = [jj['id'] for ii in graph['history'] for jj in ii['objects']]
    known_ids = list(set(known_ids))
    n_objs = len(scene['objects'])
    difference = n_objs - len(known_ids)
    diff_ids = [ii for ii in range(n_objs) if ii not in known_ids]

    # create empty objects for these
    obj_group = [{'id': ii} for ii in diff_ids]

    # create graph item
    graph_item = {'round': ques_round + 1, 'objects': obj_group,
                  'template': template['label'], 'mergeable': False}

    if 'count' in template['label']:
      graph_item['count'] = difference
      graph_item['mergeable'] = True # merge if count is known
      answer = difference
    elif 'exist' in template['label']:
      # If heads (> 0.5) -- difference > 0
      if random.random() > 0.5:
        if difference > 0:
          answer = 'yes'
        else:
          return []
      else:
        if difference == 0:
          answer = 'no'
        else:
          return []

    # no constraints, count the number of objects in true scene
    return [{'answer': answer, 'group_id': ques_round + 1,
             'objects': [], 'graph': graph_item}]

  #----------------------------------------------------------------
  elif template['label'] == 'count-all-group':
    # we need a group in the previous round
    prev_group = graph['history'][-1]
    prev_label = prev_group['template']
    if not (len(prev_group['objects']) > 1 and
            'count' not in prev_group and
            'obj-relation' not in prev_label):
      return []

    # check if count is not given before
    attrs = [ii for ii in attributes if ii in prev_group]
    count = 0
    for obj in prev_group['objects']:
      count += all([obj[ii] == prev_group['objects'][0][ii] for ii in attrs])

    # create object group
    obj_group = []
    new_obj = {'required': [], 'optional': []}
    for obj_id, ii in enumerate(scene['objects']):
      obj_copy = copy.deepcopy(new_obj)
      obj_copy['id'] = ii['id']
      obj_group.append(obj_copy)

    # create graph item
    graph_item = {'round': ques_round + 1, 'objects': copy.deepcopy(obj_group),
                  'template': template['label'],
                  'mergeable': True, 'count': count}
    # clean graph item
    graph_item = clean_graph_item(graph_item)
    # no constraints, count the number of objects in true scene
    return [{'answer': count, 'group_id': ques_round + 1,
             'objects': [], 'graph': graph_item}]
  #----------------------------------------------------------------
  elif 'count-obj-exclude' in template['label'] or \
      'exist-obj-exclude' in template['label']:
    # placeholder for object description, see below
    obj_desc = None
    prev_history = graph['history'][-1]
    scene_counts = get_attribute_counts_for_objects(scene)

    if 'imm' in template['label']:
      # we need an immediate group in the previous round
      if apply_immediate(prev_history):
        focus_id = prev_history['objects'][0]['id']
      else: return []

    elif 'early' in template['label']:
      # search through history for an object with unique attribute
      attr_counts = get_known_attribute_counts(graph)
      # get attributes with just one count
      single_count = [ii for ii, count in attr_counts.items() if count==1]
      # remove attributes that point to objects in the previous round
      # TODO: re-think this again
      obj_ids = get_unique_attribute_objects(graph, single_count)
      prev_history_obj_ids = [ii['id'] for ii in prev_history['objects']]
      single_count = [ii for ii in single_count if \
                      obj_ids[ii] not in prev_history_obj_ids]

      if len(single_count) == 0: return []

      # give preference to attributes with multiple counts in scene graph
      #scene_counts = get_attribute_counts_for_objects(scene)
      ambiguous_attrs = [ii for ii in single_count if scene_counts[ii] > 1]
      if len(ambiguous_attrs) > 0:
        focus_attr = random.choice(ambiguous_attrs)
      else:
        focus_attr = random.choice(single_count)
      focus_id = obj_ids[focus_attr]

      # unique object description
      obj_desc = {'required': [focus_attr[0]], 'optional': [],
                  focus_attr[0]: focus_attr[1]}

    # get the known attributes for the current object
    focus_obj = graph['objects'][focus_id]
    known_attrs = [attr for attr in attributes if attr in focus_obj and \
                   '%s_exclude_count' % attr not in focus_obj]


    # for count: only if existence if True, else count it trivially zero
    if 'count' in template['label']:
      for attr in known_attrs[::-1]:
        if not focus_obj.get('%s_exclude_exist' % attr, True):
          known_attrs.remove(attr)
    # for exist: get relations without exist before
    elif 'exist' in template['label']:
      known_attrs = [attr for attr in known_attrs
                     if '%s_exclude_exist' % attr not in focus_obj]

    # select an attribute
    if len(known_attrs) == 0: return[]

    # split this into zero and non-zero
    if 'exist' in template['label']:
      focus_attrs = [(ii, scene['objects'][focus_id][ii]) for ii in known_attrs]
      zero_count = [ii for ii in focus_attrs if scene_counts[ii] == 1]
      nonzero_count = [ii for ii in focus_attrs if scene_counts[ii] > 1]

      if random.random() > 0.5:
        if len(zero_count) > 0:
          attr = random.choice(zero_count)[0]
        else:
          return []
      else:
        if len(nonzero_count)>0:
          attr = random.choice(nonzero_count)[0]
        else:
          return []
    else:
      attr = random.choice(known_attrs)

    # create the object group
    obj_group = []
    new_obj = {'required': ['attribute'], 'optional': []}
    for obj in scene['objects']:
      # add if same attribute value and not focus object
      if obj[attr] == focus_obj[attr] and obj['id'] != focus_id:
        obj_copy = copy.deepcopy(new_obj)
        obj_copy['id'] = obj['id']
        obj_copy[attr] = focus_obj[attr]
        obj_group.append(obj_copy)
    answer = len(obj_group)

    ref_obj = copy.deepcopy(new_obj)
    ref_obj['id'] = focus_id
    ref_obj['volatile'] = True
    if 'exist' in template['label']:
      answer = 'yes' if answer > 0 else 'no'
      ref_obj['%s_exclude_exist' % attr] = answer
    elif 'count' in template['label']:
      ref_obj['%s_exclude_count' % attr] = answer
    obj_group.append(ref_obj)

    graph_item = {'round': ques_round+1, 'objects': copy.deepcopy(obj_group),
                  'template': template['label'], 'mergeable': True,
                  'focus_id': focus_id, 'focus_desc': obj_desc}
    if 'count' in template['label']: graph_item['count'] = answer
    graph_item = clean_graph_item(graph_item)

    ref_obj['attribute'] = attr
    return [{'answer': answer, 'group_id': ques_round + 1,
          'required': [], 'optional': [],
          'objects': [ref_obj, obj_desc], 'graph': graph_item}]
  #----------------------------------------------------------------
  elif 'count-obj-rel' in template['label'] or \
      'exist-obj-rel' in template['label']:
    # placeholder for object description, see below
    obj_desc = None
    prev_history = graph['history'][-1]

    # we need a single object in the previous round
    if 'imm2' in template['label']:
      # we need a obj-rel-imm in previous label, same as the current one
      prev_label = prev_history['template']
      cur_label = template['label']
      if 'obj-rel-imm' not in prev_label or cur_label[:5] != prev_label[:5]:
        return []
      else:
        focus_id = prev_history['focus_id']

    elif 'imm' in template['label']:
      # we need an immediate group in the previous round
      if apply_immediate(prev_history):
        focus_id = prev_history['objects'][0]['id']
      else: return []

    elif 'early' in template['label']:
      # search through history for an object with unique attribute
      attr_counts = get_known_attribute_counts(graph)

      # get attributes with just one count
      single_count = [ii for ii, count in attr_counts.items() if count==1]
      # remove attributes that point to objects in the previous round
      # TODO: re-think this again
      obj_ids = get_unique_attribute_objects(graph, single_count)
      prev_history_obj_ids = [ii['id'] for ii in prev_history['objects']]
      single_count = [ii for ii in single_count if \
                    obj_ids[ii] not in prev_history_obj_ids]

      if len(single_count) == 0: return []
      focus_attr = random.choice(single_count)

      for focus_id, obj in graph['objects'].items():
        if obj.get(focus_attr[0], None) == focus_attr[1]: break

      # unique object description
      obj_desc = {'required': [focus_attr[0]], 'optional': [],
            focus_attr[0]: focus_attr[1]}

    # get relations with unknown counts
    unknown_rels = [rel for rel in relations
            if '%s_count' % rel not in graph['objects'][focus_id]]
    # for count: only if existence if True, else count it trivially zero
    if 'count' in template['label']:
      for ii in unknown_rels[::-1]:
        if not graph['objects'][focus_id].get('%s_exist' % ii, True):
          unknown_rels.remove(ii)
    # for exist: get relations without exist before
    elif 'exist' in template['label']:
      unknown_rels = [rel for rel in unknown_rels
                      if '%s_exist' % rel not in graph['objects'][focus_id]]

    # select an object with some known objects
    if len(unknown_rels) == 0: return []

    # pick between yes/no for exist questions, 50% of times
    if 'exist' in template['label']:
      zero_count = [ii for ii in unknown_rels
                    if len(scene['relationships'][ii][focus_id]) == 0]
      nonzero_count = [ii for ii in unknown_rels
                       if len(scene['relationships'][ii][focus_id]) > 0]

      if random.random() > 0.5:
        if len(zero_count) > 0: rel = random.choice(zero_count)
        else: return []
      else:
        if len(nonzero_count) > 0: rel = random.choice(nonzero_count)
        else: return []
    else:
      rel = random.choice(unknown_rels)

    # create the object group
    obj_group = []
    new_obj = {'required': ['relation'], 'optional': []}
    obj_pool = scene['relationships'][rel][focus_id]
    for obj_id in obj_pool:
      obj_copy = copy.deepcopy(new_obj)
      obj_copy['id'] = obj_id
      obj_group.append(obj_copy)
    answer = len(obj_pool)

    ref_obj = copy.deepcopy(new_obj)
    ref_obj['id'] = focus_id
    ref_obj['volatile'] = True
    if 'exist' in template['label']:
      answer = 'yes' if answer > 0 else 'no'
      ref_obj['%s_exist' % rel] = answer
    elif 'count' in template['label']:
      ref_obj['%s_count' % rel] = answer
    obj_group.append(ref_obj)

    graph_item = {'round': ques_round+1, 'objects': copy.deepcopy(obj_group),
                  'template': template['label'], 'mergeable': True,
                  'focus_id': focus_id, 'focus_desc': obj_desc}
    if 'count' in template['label']: graph_item['count'] = answer
    graph_item = clean_graph_item(graph_item)

    #ref_obj['relation'] = rel
    # add attribute as argument
    return [{'answer': answer, 'group_id': ques_round + 1,
          'required': [], 'optional': [], 'relation': rel,
          'objects': [ref_obj, obj_desc], 'graph': graph_item}]
  #----------------------------------------------------------------
  elif 'count-attribute' in template['label'] or \
      'exist-attribute' in template['label']:
    if 'group' in template['label']:
      # we need an immediate group in the previous round
      prev_history = graph['history'][-1]
      prev_label = prev_history['template']

      # if exist: > 0 is good, else > 1 is needed
      min_count = 0 if 'exist' in prev_label else 1
      if (len(prev_history['objects']) > min_count and
          prev_history['mergeable'] and
          'obj-relation' not in prev_label):
        obj_pool = graph['history'][-1]['objects']
      else:
        return []
    else: obj_pool = scene['objects']

    # get counts for attributes, and sample evenly with 0 and other numbers
    counts = get_attribute_counts_for_objects(scene, obj_pool)

    # if exist, choose between zero and others wiht 0.5 probability
    zero_prob = 0.5 if 'exist' in template['label'] else 0.7
    if random.random() > zero_prob:
      pool = [ii for ii in counts if counts[ii] == 0]
    else: pool = [ii for ii in counts if counts[ii] != 0]

    # check if count is already known
    attr_pool = filter_attributes_with_known_counts(graph, pool)

    # for exist: get known attributes and remove them
    if 'exist' in template['label']:
      known_attr = get_known_attributes(graph)
      attr_pool = [ii for ii in attr_pool if ii not in known_attr]

    # if non-empty, sample it
    if len(attr_pool) == 0: return []

    attr, value = random.choice(attr_pool)
    # add a hypothesi, and return the answer
    count = 0
    obj_group = []
    new_obj = {attr: value, 'required': [attr], 'optional': []}
    for index, obj in enumerate(obj_pool):
      if scene['objects'][obj['id']][attr] == value:
        obj_copy = copy.deepcopy(new_obj)
        obj_copy['id'] = obj['id']
        obj_group.append(obj_copy)
        count += 1

    graph_item = {'round': ques_round + 1, 'objects': copy.deepcopy(obj_group),
            'template': template['label'],
            'mergeable': True, attr: value}

    if 'count' in template['label']:
      graph_item['count'] = count
      answer = count
    elif 'exist' in template['label']:
      answer = 'yes' if count > 0 else 'no'
    # clean graph item
    graph_item = clean_graph_item(graph_item)
    if count == 0:
      # fake object group, to serve for arguments
      obj_group = [{attr: value, 'required': [attr], 'optional': []}]

    return [{'answer': answer, 'group_id': ques_round + 1,
             'required': [attr], 'optional': [],
             'count': 9999, 'objects': obj_group,
             'graph': graph_item}]
  #----------------------------------------------------------------
  elif 'seek-attr-rel' in template['label']:
    # placeholder for object description, see below
    obj_desc = None
    prev_history = graph['history'][-1]

    if 'imm' in template['label']:
      # we need an immediate group in the previous round
      if apply_immediate(prev_history):
        focus_id = prev_history['objects'][0]['id']
      else: return []

    elif 'early' in template['label']:
      # search through history for an object with unique attribute
      attr_counts = get_known_attribute_counts(graph)

      # get attributes with just one count
      single_count = [ii for ii, count in attr_counts.items() if count==1]
      # remove attributes that point to objects in the previous round
      # TODO: re-think this again
      obj_ids = get_unique_attribute_objects(graph, single_count)
      prev_history_obj_ids = [ii['id'] for ii in prev_history['objects']]
      single_count = [ii for ii in single_count if \
                    obj_ids[ii] not in prev_history_obj_ids]
      if len(single_count) == 0: return []

      # give preference to attributes with multiple counts in scene graph
      scene_counts = get_attribute_counts_for_objects(scene)
      ambiguous_attrs = [ii for ii in single_count if scene_counts[ii] > 1]
      if len(ambiguous_attrs) > 0:
        focus_attr = random.choice(ambiguous_attrs)
      else:
        focus_attr = random.choice(single_count)
      focus_id = obj_ids[focus_attr]

      # unique object description
      obj_desc = {'required': [focus_attr[0]], 'optional': [],
            focus_attr[0]: focus_attr[1]}

    # for each relation, get the object, sample an attribute, and sample
    hypotheses = []
    for rel in relations:
      gt_relations = scene['relationships'][rel]
      objs = [(ii, len(gt_relations[ii])) for ii in gt_relations[focus_id]]
      objs = sorted(objs, key=lambda x:x[1], reverse=True)
      if len(objs) == 0:
        # add a null hypotheses
        # check if the object is known to be extreme
        if '%s_count' % rel not in graph['objects'][focus_id] \
          and '%s_exist' % rel not in graph['objects'][focus_id]:
          hypotheses.append((None, rel, random.choice(attributes)))
        continue

      closest_obj = objs[0][0]
      # check what attributes are known/unknown
      known_info = graph['objects'].get(closest_obj, {})
      for attr in attributes:
        if attr not in known_info:
          hypotheses.append((closest_obj, rel, attr))

    if len(hypotheses) == 0: return []
    sample_id, rel, attr = random.choice(hypotheses)
    # add the new attribute to object
    new_obj = {'required': ['attribute', 'relation'],
          'optional': [], 'id': sample_id}

    if sample_id is not None: answer = scene['objects'][sample_id][attr]
    else: answer = 'none'
    new_obj[attr] = answer

    graph_item = {'round': ques_round+1, 'objects': [copy.deepcopy(new_obj)],
            'template': template['label'], 'mergeable': True,
            'focus_id': focus_id, 'focus_desc': obj_desc}
    # remove objects if none
    if sample_id is None: graph_item['objects'] = []
    graph_item = clean_graph_item(graph_item)

    # add attribute as argument
    new_obj['attribute'] = attr
    #new_obj['relation'] = rel
    return [{'answer': new_obj[attr], 'group_id': ques_round + 1,
          'required': [], 'optional': [], 'relation': rel,
          'objects': [new_obj, obj_desc], 'graph': graph_item}]
  #----------------------------------------------------------------
  elif 'seek-attr' in template['label']:
    # placeholder for object description, see below
    obj_desc = None
    prev_history = graph['history'][-1]
    prev_label = prev_history['template']
    implicit_attr = None

    # we need a single object in the previous round
    if 'imm2' in template['label']:
      # we need a seek-attr-imm/seek-attr-rel-imm in previous label
      if 'seek-attr-imm' not in prev_label \
            and 'seek-attr-rel-imm' not in prev_label:
        return []
      elif len(prev_history['objects']) == 0: return []
      else: focus_id = prev_history['objects'][0]['id']

    elif 'imm' in template['label']:
      # we need an immediate group in the previous round
      if apply_immediate(prev_history):
        focus_id = prev_history['objects'][0]['id']
      else: return []

    elif 'sim' in template['label']:
      if 'seek-attr-imm' not in prev_label: return[]
      else:
        prev_obj = prev_history['objects'][0]
        focus_id = prev_obj['id']
        attr = [ii for ii in attributes if ii in prev_obj]
        assert len(attr) == 1, 'Something wrong in previous history!'
        implicit_attr = attr[0]

    if 'early' in template['label']:
      # search through history for an object with unique attribute
      attr_counts = get_known_attribute_counts(graph)

      # get attributes with just one count
      single_count = [ii for ii, count in attr_counts.items() if count==1]
      # remove attributes that point to objects in the previous round
      # TODO: re-think this again
      obj_ids = get_unique_attribute_objects(graph, single_count)
      prev_history_obj_ids = [ii['id'] for ii in prev_history['objects']]
      single_count = [ii for ii in single_count if \
                    obj_ids[ii] not in prev_history_obj_ids]

      # if there is an attribute, eliminate those options
      if implicit_attr is not None:
        single_count = [ii for ii in single_count if ii[0]!=implicit_attr]
        obj_ids = get_unique_attribute_objects(graph, single_count)

        # again rule out objects whose implicit_attr is known
        single_count = [ii for ii in single_count \
                  if implicit_attr not in \
                    graph['objects'][obj_ids[ii]]]

      if len(single_count) == 0: return []

      # give preference to attributes with multiple counts in scene graph
      scene_counts = get_attribute_counts_for_objects(scene)
      ambiguous_attrs = [ii for ii in single_count if scene_counts[ii] > 1]
      if len(ambiguous_attrs) > 0:
        focus_attr = random.choice(ambiguous_attrs)
      else:
        focus_attr = random.choice(single_count)
      focus_id = get_unique_attribute_objects(graph, [focus_attr])[focus_attr]

      # unique object description
      obj_desc = {'required': [focus_attr[0]], 'optional': [],
                  focus_attr[0]: focus_attr[1]}

    # get unknown attributes, randomly sample one
    if implicit_attr is None:
      unknown_attrs = [attr for attr in attributes
              if attr not in graph['objects'][focus_id]]

      # TODO: select an object with some known objects
      if len(unknown_attrs) == 0: return []
      attr = random.choice(unknown_attrs)
    else: attr = implicit_attr

    # add the new attribute to object
    new_obj = {'required': ['attribute'], 'optional': [], 'id': focus_id}
    if 'sim' in template['label']: new_obj['required'] = []
    new_obj[attr] = scene['objects'][focus_id][attr]

    graph_item = {'round': ques_round+1, 'objects': [copy.deepcopy(new_obj)],
            'template': template['label'], 'mergeable': True,
            'focus_id': focus_id, 'focus_desc': obj_desc}
    graph_item = clean_graph_item(graph_item)

    # add attribute as argument
    new_obj['attribute'] = attr
    return [{'answer': new_obj[attr], 'group_id': ques_round + 1,
          'required': [], 'optional': [],
          'objects': [new_obj, obj_desc], 'graph': graph_item}]

  return []


# get a list of known objects
def get_known_objects(graph):
  pass


# get a list of known attributes
def get_known_attributes(graph):
  known_attrs = []
  for obj_id, obj_info in graph['objects'].items():
    # the attribute is unique already
    #if obj_info.get('unique', False): continue

    for attr in attributes:
      if attr in obj_info:
        known_attrs.append((attr, obj_info[attr]))

  # also go over the groups
  for ii in graph['history']:
    # a group of objects, with unknown count
    #if 'count' not in ii: continue

    for attr in attributes:
      if attr in ii: known_attrs.append((attr, ii[attr]))
  known_attrs = list(set(known_attrs))

  return known_attrs


# first get known attributes
def get_known_attribute_counts(graph):
  known_attrs = get_known_attributes(graph)

  # go through objects and count
  counts = {ii:0 for ii in known_attrs}
  for _, obj in graph['objects'].items():
    for attr, val in known_attrs:
      if obj.get(attr, None) == val: counts[(attr, val)] += 1

  return counts


# check if count is already known
def filter_attributes_with_known_counts(graph, known_attrs):
  for attr, val in known_attrs[::-1]:
    for ii in graph['history']:
      # a group of objects, with unknown count
      if 'count' not in ii: continue

      # count is absent
      if ii.get(attr, None) == val:
        known_attrs.remove((attr, val))

  return known_attrs


def clean_graph_item(graph_item):
  """Method to clean up graph item (remove 'required' and 'optional' tags).

  Args:
    graph_item: Input graph item to be cleaned.

  Returns:
    clean_graph_item: Copy of the graph item after cleaning.
  """

  clean_graph_item = copy.deepcopy(graph_item)
  if 'optional' in clean_graph_item:
    del clean_graph_item['optional']
  if 'required' in clean_graph_item:
    del clean_graph_item['required']

  for index, ii in enumerate(clean_graph_item['objects']):
    if 'optional' in ii:
      del clean_graph_item['objects'][index]['optional']
    if 'required' in ii:
      del clean_graph_item['objects'][index]['required']

  return clean_graph_item


def get_attribute_counts_for_objects(scene, objects=None):
  # initialize the dictionary
  counts = {}
  for attr, vals in values.items():
    for val in vals: counts[(attr, val)] = 0

  # now count for each given object
  if objects is None: objects = scene['objects']
  for obj in objects:
    for attr in attributes:
      key = (attr, scene['objects'][obj['id']][attr])
      counts[key] = counts.get(key, 0) + 1

  return counts


def get_unique_attribute_objects(graph, uniq_attrs):
  obj_ids = {}
  for obj_id, obj in graph['objects'].items():
    for attr, val in uniq_attrs:
      if obj.get(attr, '') == val:
        # at this point the key should not be present
        assert (attr, val) not in obj_ids, 'Attributes not unique!'
        obj_ids[(attr, val)] = obj_id

  return obj_ids


def sample_optional_tags(optional, sample_probs):
  """Sample additional tags depending on given sample probabilities.

  Args:
    optional: List of optional tags to sample from.
    sample_probs: Probabilities of sampling 'n' tags.
  """

  sampled = []
  if len(optional) > 0:
    n_sample = np.random.choice([0, 1], 1, p=sample_probs)[0]
    n_sample = min(n_sample, len(optional))
    sampled = random.sample(optional, n_sample)
  return sampled
