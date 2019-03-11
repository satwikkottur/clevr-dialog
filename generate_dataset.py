r"""Generates CLEVR-Dialog dataset.

Needs access to the following files:
synonyms: Contains several synonyms for each word in the question/caption.
caption templates: List of caption templates.
question templates: List of question templates.
metainfo: Meta-information related to attributes and values of CLEVR objects.

Usage:
 python -u generate_dataset.py \
   --scene_path="data/scenes/CLEVR_train_scenes.json" \
   --num_beams=100 \
   --num_workers=12 \
   --save_path="data/clevr_train_raw.json"

Author: Satwik Kottur
"""


import copy
import collections
import json
import multiprocessing
import os
import random
import re
import time
from absl import flags
from absl import app
import numpy as np
from tqdm import tqdm as progressbar

import clevr_utils as utils
import global_vars as gvars
import constraints


FLAGS = flags.FLAGS
flags.DEFINE_string('synonym_path', 'templates/synonyms.json',
                    'Path to synonyms file')
flags.DEFINE_string('metainfo_path', 'templates/metainfo.json',
                    'Path to meta information file')
flags.DEFINE_string('caption_template_root', 'templates/captions/',
                    'Root to folder with caption templates')
flags.DEFINE_string('question_template_root', 'templates/questions/',
                    'Root to folder with question templates')
flags.DEFINE_string('scene_path', 'scenes/CLEVR_train_scenes.json',
                    'Path to CLEVR scene path json file')
flags.DEFINE_string('scene_id_file', '',
                    'Path to specific CLEVR scene ids to generate dialogs')
flags.DEFINE_string('save_path', 'dialogs_gen_debug.json',
                    'Path to save the dataset json')
flags.DEFINE_integer('num_beams', 100, 'Number of beams in dialog search')
flags.DEFINE_integer('num_workers', 1, 'Number of workers to use in search')
flags.DEFINE_integer('captions_per_image', 5, 'Number of captions per image')
flags.DEFINE_integer('num_images', -1,
                     'Number of images to generate dialogs. -1 for all.')
flags.DEFINE_integer('num_rounds', 10, 'Number of rounds in each dialog')


# Number of beams and distribution of question types.
# Start cutting down beams after 5th round.
# Heuristics (for round 4):
# A. count <= 2  1 <= seek <= 3  exist <= 2
# B. count + exist <= 3
# C. Independent questions <= 1
# Heuristics (for round 5):
# A. count <= 2  2 <= seek <= 4  exist <= 2
# B. count + exist <= 3
# C. Independent questions <= 1
ranges = {3: {'indep': [0, 1], 'seek': [1, 4], 'exist': [0, 1],
              'count': [0, 1], 'exist+count': [0, 2]},
          4: {'indep': [0, 1], 'seek': [2, 4], 'exist': [0, 1],
              'count': [0, 1], 'exist+count': [0, 2]},
          5: {'indep': [0, 1], 'seek': [2, 5], 'exist': [0, 2],
              'count': [0, 2], 'exist+count': [0, 3]},
          6: {'indep': [0, 1], 'seek': [2, 5], 'exist': [0, 2],
              'count': [0, 2], 'exist+count': [0, 3]},
          7: {'indep': [0, 2], 'seek': [3, 5], 'exist': [0, 2],
              'count': [0, 2], 'exist+count': [0, 3]},
          8: {'indep': [0, 2], 'seek': [3, 6], 'exist': [0, 3],
              'count': [0, 3], 'exist+count': [0, 3]},
          9: {'indep': [0, 2], 'seek': [3, 6], 'exist': [0, 3],
              'count': [0, 3], 'exist+count': [0, 4]}}


def mapping(tag):
  """Maps tag to attribute.

  Args:
    tag: An input tag

  Returns:
    tag_label: Label for the input tag
  """

  return gvars.METAINFO['tag_map'][tag.replace('1', '')]


def inv_mapping(attribute, arg_id=0):
  """Inverse maps attribute to tag.

  Args:
    attribute: Name of the attribute
    arg_id: Argument id to use. Append 1 if arg_id is 1, else nothing

  Returns:
    base_tag: The string for the tag
  """

  base_tag = gvars.METAINFO['tag_inv_map'][attribute]
  if arg_id > 0:
    base_tag = base_tag[:-1] + str(arg_id) + base_tag[-1]

  return base_tag


def get_tag_group(tag):
  """Gets the group id from tag string.

  For example, tag string of <S> is 0, <S1> is 1.
  Assumes single digit group id.

  Args:
    tag: Tag string

  Returns:
    group_id: Return extracted group id
  """

  group_id = 0 if len(tag) <= 3 else int(tag[-2])
  return group_id


def replace_attribute(text, tag, obj_group, eliminate=False):
  """Replaces the attribute tags in text using available object properties.

  NOTE: If shape is to be replaced, we use 'thing' in its place.

  Args:
    text: The text template to perform replacement
    tag: The tags to replace in the text
    obj_group: Available object properties to replace with
    eliminate: Eliminate the remaining attribute tags

  Returns:
    replaced_text: The replaced text
  """

  group = get_tag_group(tag)
  if mapping(tag) == 'relation':
    # Actual relation tag, else position tag.
    if tag == '<R>':
      relation_list = gvars.METAINFO['relation_phrases'][obj_group['relation']]
      relation_cand = random.choice(relation_list)
    else:
      relation_cand = obj_group['relation']

    return text.replace(tag, relation_cand)

  if mapping(tag) == 'shape':
    if eliminate:
      replacer = 'thing'
    else:
      replacer = str(obj_group['objects'][group][mapping(tag)])

    # Plural forms for groups.
    if obj_group.get('count', 1) > 1 or obj_group.get('use_plural', False):
      replacer += 's'
  elif mapping(tag) == 'count':
    if eliminate:
      replacer = ''
    else:
      replacer = str(obj_group['count'])
  else:
    if eliminate:
      replacer = ''
    else:
      replacer = str(obj_group['objects'][group][mapping(tag)])
  return text.replace(tag, replacer)


def realize_text_and_extract_scene(scene, template, filter_objs):
  """Samples attributes for template using filtered objects.

  In addition, creates scene graph for the new information added.

  Args:
    scene: Current scene graph
    template: Text template to use to generate questions
    filter_objs: Set of objects satisfying constraints of current template

  Returns:
    sample: Contains the text realization and scene graph
  """

  default_list = lambda: collections.defaultdict(list)
  graph = {'relationships': collections.defaultdict(default_list),
           'counts': {}, 'exists': {}, 'history': [], 'objects': {}}

  # number of inputs
  n_inputs = template.get('inputs', 1)
  # sample a text template
  text_sample = random.choice(template['text'])
  text_sample_index = template['text'].index(text_sample)

  # extract attribute tags and get them into groups
  tags = re.findall('(<[\d\w]*>)', text_sample)

  tag_groups = collections.defaultdict(list)
  for tag in tags:
    group_id = get_tag_group(tag)
    tag_groups[group_id].append(tag)

  # sample a random element from filtered
  arg_sample = random.choice(filter_objs)
  # scene information obtained from the current round
  graph_item = arg_sample['graph']

  # remove tags from text not allowed by filter_objs
  for arg_ind in range(n_inputs):
    obj_sample = arg_sample['objects'][arg_ind]
    avail_attrs = obj_sample['optional'] + obj_sample['required']

    for ii in tag_groups[arg_ind][::-1]:
      if mapping(ii) not in avail_attrs:
        tag_groups[arg_ind].remove(ii)
        text_sample = replace_attribute(text_sample, ii, arg_sample, True)

    # assert that all required attributes are present as tags
    for attribute in obj_sample['required']:
      required_tag = inv_mapping(attribute, arg_ind)
      assert required_tag in tag_groups[arg_ind], \
          'A required attribute is missing in template!'

    # start compiling tags to keep
    tags_to_keep = [inv_mapping(ii, arg_ind) for ii in obj_sample['required']]

    # filter out those not present in text template
    optional_tags = [inv_mapping(ii,arg_ind) for ii in obj_sample['optional']]
    optional_tags = [ii for ii in optional_tags if ii in tag_groups[arg_ind]]

    # if tags_to_keep is empty, sample from optional with 1:70 2:25  3:5
    if len(optional_tags) > 0:
      if len(tags_to_keep) > 0:
        n_tags_sample = [0, 1, 2]
      else: n_tags_sample = [1, 2, 3]
      n_sample = np.random.choice(n_tags_sample, 1,
                                  p=gvars.METAINFO['probabilities'],
                                  replace=False)
      # lower cap at the length of optional
      n_sample = min(n_sample[0], len(optional_tags))
      if n_sample > 0:
        tags_to_keep += random.sample(optional_tags, n_sample)

    # now create a dictionary of placeholders with actual attribute values
    for tag in tag_groups[arg_ind]:
      remove = tag not in tags_to_keep
      text_sample = replace_attribute(text_sample, tag, arg_sample, remove)

    # remove attributes from objects not included in tags_to_keep
    if 'objects' in graph_item:
      for ii in gvars.METAINFO['attributes']:
        if inv_mapping(ii, arg_ind) not in tags_to_keep:
          if ii in graph_item['objects'][arg_ind]:
            del graph_item['objects'][arg_ind][ii]

  # record the caption info
  graph_item['round'] = 0

  sample = {}
  sample['template_info'] = [copy.deepcopy(template)]
  del sample['template_info'][-1]['text']
  sample['template_info'][-1]['index'] = text_sample_index
  sample['caption'] = text_sample
  sample['dialog'] = []

  # append history, update scene graph, and save the new scene graph
  graph['history'].append(graph_item)
  sample['graph'] = utils.merge_update_scene_graph(graph, graph_item)
  return sample


def realize_question(dialog, template, filter_objs):
  """Samples attributes for template using filtered objects.

  In addition, creates scene graph for the new information added.

  Args:
    scene: Current scene graph
    template: Text template to use to generate questions
    filter_objs: Set of objects satisfying constraints of current template

  Returns:
    sample: Contains the text realization and scene graph
  """

  # Number of inputs.
  n_inputs = template.get('inputs', 0)
  # Sample a text template.
  text_sample = random.choice(template['text'])
  text_sample_index = template['text'].index(text_sample)

  # Extract attribute tags and get them into groups.
  tags = re.findall('(<[\d\w]*>)', text_sample)
  tag_groups = collections.defaultdict(list)
  for tag in tags:
    group_id = get_tag_group(tag)
    tag_groups[group_id].append(tag)

  # Sample a random element from filtered.
  arg_sample = random.choice(filter_objs)

  # Remove tags from text not allowed by filter_objs.
  for arg_ind in range(n_inputs):
    obj_sample = arg_sample['objects'][arg_ind]
    avail_attrs = obj_sample['optional'] + obj_sample['required']

    for ii in tag_groups[arg_ind][::-1]:
      if mapping(ii) not in avail_attrs:
        tag_groups[arg_ind].remove(ii)
        text_sample = replace_attribute(text_sample, ii, arg_sample, True)

    # Assert that all required attributes are present as tags.
    for attribute in obj_sample['required']:
      required_tag = inv_mapping(attribute, arg_ind)
      # Make an exception for <R> and <P>
      if required_tag == '<R>' and '<P>' in tag_groups[arg_ind]:
        continue
      assert required_tag in tag_groups[arg_ind], \
        'A required attribute is missing in template!'

    # Start compiling tags to keep.
    tags_to_keep = [inv_mapping(ii, arg_ind) for ii in obj_sample['required']]
    # Filter out those not present in text template.
    optional_tags = [inv_mapping(ii,arg_ind) for ii in obj_sample['optional']]
    optional_tags = [ii for ii in optional_tags if ii in tag_groups[arg_ind]]

    # If tags_to_keep is empty, sample from optional with (1:70, 2:25, 3:5).
    if len(optional_tags) > 0:
      if len(tags_to_keep) > 0:
        n_tags_sample = [0, 1, 2]
      else:
        n_tags_sample = [1, 2, 3]
      n_sample = np.random.choice(n_tags_sample, 1,
                                  p=gvars.METAINFO['probabilities'],
                                  replace=False)
      # Lower cap at the length of optional.
      n_sample = min(n_sample[0], len(optional_tags))
      if n_sample > 0:
        tags_to_keep += random.sample(optional_tags, n_sample)

    # Now create a dictionary of placeholders with actual attribute values.
    for tag in tag_groups[arg_ind]:
      remove = tag not in tags_to_keep
      text_sample = replace_attribute(text_sample, tag, arg_sample, remove)

  # Record info and merge scene graphs.
  dialog_datum = {'question': text_sample, 'answer': arg_sample['answer'],
                  'template': template['label']}
  dialog['template_info'].append(template.copy())
  del dialog['template_info'][-1]['text']
  dialog['template_info'][-1]['index'] = text_sample_index

  dialog['dialog'].append(dialog_datum)
  graph_item = arg_sample['graph']

  # If mergeable, add it to the objects list.
  dialog['graph'] = utils.merge_update_scene_graph(dialog['graph'], graph_item)

  # If there are volatile objects in the graph item, remove them.
  for obj in graph_item['objects'][::-1]:
    if obj.get('volatile', False): graph_item['objects'].remove(obj)
  dialog['graph']['history'].append(graph_item)
  return dialog


def clean_text_subroutine(text, thing, suffix):
  """Cleans the text and substitutes thing with object (subroutine).

  Args:
    text: Text string to be cleaned
    thing: Whether to use 'thing' or 'object'
    suffix: Either '?' (question) or '.' (caption)

  Returns:
    clean_text: Text string after cleaning procedure
  """

  # Synonyms + skipping optional part of the sentence
  clean_text = skip_and_replace_phrases(text)

  # Remove full stop, empty spaces, capitalize the start letter.
  clean_text = re.sub(' +', ' ', clean_text.replace(suffix, '').strip(' '))
  # First replace 'a thing' -> 'an object'.
  # Then perform remaining actions.
  if thing == 'object':
    clean_text = clean_text.replace('a thing', 'an object')
  clean_text = clean_text.replace('thing', thing)
  clean_text = clean_text[0].upper() + clean_text[1:] + suffix
  return clean_text


def clean_dialog_text(dialogs):
  """Cleans the dialog texts.

  Args:
    dialogs: Generated dialogs to perform text cleaning

  Returns:
    dialogs: Return the dialogs after cleaning the text inplace
  """

  # Replace thing with object throughout with probability 0.5.
  thing = 'thing' if random.random() > 0.5 else 'object'
  for index, dialog_datum in enumerate(dialogs):
    # Clean the caption.
    text = dialog_datum['caption']
    dialogs[index]['caption'] = clean_text_subroutine(text, thing, '.')

    for r_id, dialog in enumerate(dialog_datum['dialog']):
      # Clean the question.
      text = dialog['question']
      text = clean_text_subroutine(text, thing, '?')
      dialogs[index]['dialog'][r_id]['question'] = text
  return dialogs


def skip_and_replace_phrases(text):
  """Substitutes synonyms and skips optional parts stochastically.

  Args:
    text: Text string

  Returns:
    text: Text string with synonyms replaced and optional parts skipped
  """

  # For each text in [], replace it with '' with probability 0.5.
  matches = re.findall('(\[[ \w]*\])', text)
  for match in matches:
    if random.uniform(0, 1) > 0.5:
      text = text.replace(match, '')
    else:
      text = text.replace(match, match[1:-1])

  # Remove empty spaces, if any.
  text = re.sub(' +', ' ', text)
  # Search for synonyms, replace at uniformly random.
  text = text.lower()
  for key, values in gvars.METAINFO['synonym_keys']:
    if key in text:
      text = text.replace(key, random.choice(values))
  return text


def generate_captions(scenes, templates):
  """Wrapper generates captions.

  Args:
    scenes: List of scene graphs for which to generate captions
    templates: List of available caption templates

  Returns:
    generated_content: Captions generated for the input scenes
  """

  template_dictionary = {ii['label']: ii for ii in templates}
  generated_content = []
  for scene in scenes['scenes'][0:FLAGS.num_images]:
    content = {}
    # Copy over image_index, split, image_filename from scene.
    for key in ['image_index', 'split', 'image_filename']:
      content[key] = scene[key]

    content['dialogs'] = []
    # Filter objects based on constraints.
    filter_objs = constraints.caption(scene, templates)
    for filter_obj in filter_objs:
      # Realize the text, and return the partial scene knowledge (q).
      template = template_dictionary[filter_obj[0]['graph']['template']]
      sample = realize_text_and_extract_scene(scene, template, filter_obj)
      # Add it to the list of dialogs.
      content['dialogs'].append(sample)
    generated_content.append(content)
  return generated_content


def generate_questions(scenes, dialogs, templates, params):
  """Wrapper generates questions.

  Args:
    scenes: List of scene graphs to generate questions
    dialogs: Contains already generated captions for scenes graphs
    templates: List of available question templates
    params: Beam search parameters for question generation

  Returns:
    new_dialogs: Generated raw dialogs with captions and questions
  """

  new_dialogs = []
  for scene_id, dialog_datum in enumerate(dialogs):
    image_dialogs = copy.deepcopy(dialog_datum)
    image_dialogs['dialogs'] = []

    for dialog in dialog_datum['dialogs']:
      # Pick a template at random.
      flag = False
      iter_count = 0
      while not flag:
        # Pick a template at random.
        template = random.choice(templates)

        # Filter objects based on constraints.
        filter_objs = constraints.question(scenes['scenes'][scene_id],
                                           dialog, template)
        flag = len(filter_objs) != 0

        # Extreme case -- exit
        iter_count += 1
        if iter_count > 10:
          break

      # Realize q question.
      if flag:
        deep_copy = copy.deepcopy(dialog)
        gen_dialog = realize_question(deep_copy, template, filter_objs)
        image_dialogs['dialogs'].append(copy.deepcopy(gen_dialog))
    new_dialogs.append(image_dialogs)

  return new_dialogs


def worker(scenes, cap_templates, ques_templates, worker_id, out_q):
  """Worker method generates dialogs (caption + questions) for pool of scenes.

  Args:
    scenes: List of CLEVR scenes to generate dialogs
    cap_templates: Templates for caption generation
    ques_templates: Templates for question generation
    worker_id: Id for the current worker
    out_q: Output queue to save generated dialogs from different sources

  Returns:
    Adds dialogs against the worker id in the output queue.
  """

  dialogs = []
  for index, scene in enumerate(scenes):
    cur_time = time.strftime('%a-%d%b%y-%X', time.gmtime())
    print('Generating [ %s ] [ Worker: %d, Progress: %d/%d Scene:  %d ]' % \
          (cur_time, worker_id, index, len(scenes), scene['image_index']))
    try:
      gen_dialog = generate_dialog_bfs(scene, cap_templates, ques_templates)
      dialogs.append(json.loads(json.dumps(gen_dialog)))
    except:
      print('NOTE: Missing data for %d' % scene['image_index'])
  out_q.put({worker_id: dialogs})


def generate_dialog_bfs(scene, cap_templates, ques_templates):
  """Perform approximate breadth-first-search (BFS) to generate dialogs.

  Args:
    scene: Scene graph for the CLEVR image
    cap_templates: List of caption templates
    ques_templates: List of question templates

  Returns:
    bundle: List of dialogs generated for the input scene graph
  """

  bundle = {}
  # Generate captions for the scene.
  # Copy over image_index, split, image_filename from scene.
  for key in ['image_index', 'split', 'image_filename']:
    bundle[key] = scene[key]

  template_dictionary = {ii['label']: ii for ii in cap_templates}
  content = {}

  # Filter objects based on constraints on captions.
  filter_objs = constraints.caption(scene, cap_templates)

  for filter_obj in filter_objs:
    # Realize the text, and return the partial scene knowledge (q).
    template = template_dictionary[filter_obj[0]['graph']['template']]
    sample = realize_text_and_extract_scene(scene, template, filter_obj)
    # Add it to the list of dialogs.
    content[template['label']] = [sample]

  # Now generate questions.
  # Group templates, exist/count of similar type together.
  ques_groups = collections.defaultdict(list)

  labels = [ii['label'] for ii in ques_templates]
  #print('\n'.join(labels))
  for index, ii in enumerate(ques_templates):
    if 'exist' in ii['label'] or 'count' in ii['label']:
      ques_groups[labels[index][4:]].append(ii)
    else:
      ques_groups[labels[index]].append(ii)

  for round_id in range(FLAGS.num_rounds):
    new_content = {}

    # For each group.
    for cap_label, cap_dialogs in content.items():
      cur_pool = []
      for dialog_datum in cap_dialogs:
        for _, group in ques_groups.items():
          template = random.choice(group)
          # Make a copy.
          datum_copy = copy.deepcopy(dialog_datum)

          # Filter objects based on constraints.
          filter_objs = constraints.question(scene, datum_copy, template)
          if len(filter_objs) == 0: continue

          # Realize q question.
          gen_dialog = realize_question(datum_copy, template, filter_objs)
          cur_pool.append(gen_dialog)

      if round_id in ranges:
        for d_id, dialog in enumerate(cur_pool):
          n_types = {'indep': 0, 'seek': 0, 'exist': 0, 'count': 0}
          keep_dialog = True

          labels = [ii['label'] for ii in dialog['template_info'][1:]]
          for label in labels:
            if label in gvars.METAINFO['independent_questions']:
              n_types['indep'] += 1

            label_type = label.split('-')[0]
            n_types[label_type] += 1

          # Heuristic A, C
          for q_type, count in n_types.items():
            limit = ranges[round_id][q_type]
            if limit[0] > count or count > limit[1]:
              keep_dialog = False
              break

          # Heuristic B
          limit = ranges[round_id]['exist+count']
          if n_types['count'] + n_types['exist'] > limit[1]:
            keep_dialog = False
          if not keep_dialog: cur_pool[d_id] = None
        cur_pool = [ii for ii in cur_pool if ii is not None]

      # Keep limited number of beams (for speed).
      if len(cur_pool) > FLAGS.num_beams:
        cur_pool = sample_beams(cur_pool)[:FLAGS.num_beams]
      new_content[cap_label] = cur_pool
    content = copy.deepcopy(new_content)

  # Get dialogs with sim, imm2, early questions.
  for cap_label, cap_dialogs in content.items():
    # Sample beams.
    content[cap_label] = sample_beams(cap_dialogs)

  # Remove keys that are empty.
  empty_keys = [key for key, val in content.items() if len(val) == 0]
  for key in empty_keys:
    del content[key]

  # For each caption, sample one.
  sampled_dialogs = []
  for cap_label, cap_dialogs in content.items():
    if len(cap_dialogs) > 0:
      sampled_dialogs.append(cap_dialogs.pop())

  # Get 5 per image, compensate by taking from other entries.
  content_keys = [ii for ii in content.keys()]
  while len(sampled_dialogs) < 5:
    random_label = random.choice(content_keys)
    sampled_dialogs.append(cap_dialogs.pop())

  # Finally, make the dialog text readable.
  sampled_dialogs = clean_dialog_text(sampled_dialogs)

  # Generate the coreference chain.
  for dialog_id, dialog in enumerate(sampled_dialogs):
    sampled_dialogs[dialog_id] = identify_coref_chains(dialog)
  bundle['dialogs'] = sampled_dialogs
  return bundle


def sample_beams(dialogs):
  """Samples beams based on the number of constraints satisfied.

  Args:
    dialogs: Generated dialogs to sample beams

  Returns:
    sampled_dialogs: List of sampled dialogs based on the constraints
  """

  num_constraints = []
  for d_id, dialog in enumerate(dialogs):
    satisfied = 0
    labels = [ii['label'] for ii in dialog['template_info'][1:]]

    # Have a imm2 for sure
    satisfied += np.sum(['imm2' in ii for ii in labels])
    # Have a imm2 for sure
    satisfied += np.sum(['sim' in ii for ii in labels])
    # Have 'early'
    satisfied += min(4, np.sum(['early' in ii for ii in labels]))

    # Add it with the number of constraints it satisfies.
    num_constraints.append((satisfied, d_id))

  # Then order.
  sort_key = lambda x: (x[0], random.random())
  ids = sorted(num_constraints, key=sort_key, reverse=True)
  sampled_dialogs = [dialogs[ii[1]] for ii in ids]
  return sampled_dialogs


def identify_coref_chains(dialog):
  """Identifies the coreference chains in generated dialog.

  Args:
    dialog: Generated dialogs for which coreference chains to be identified

  Returns:
    dialog: A copy of dialog, with coreference chains annotated
  """

  for r_id, datum in enumerate(dialog['dialog']):
    label = datum['template']
    if label in gvars.METAINFO['independent_questions']:
      dialog['graph']['history'][r_id + 1]['dependence'] = None
      continue

    if (label == 'exist-attribute-group' or label == 'count-attribute-group' or
        label == 'count-all-group'):
      dialog['graph']['history'][r_id + 1]['dependence'] = r_id - 1
      continue

    if 'imm' in label:
      dialog['graph']['history'][r_id + 1]['dependence'] = r_id - 1
      continue

    if 'early' in label:
      # Go over previous history.
      cur_history  = dialog['graph']['history'][r_id + 1]
      assert 'focus_id' in cur_history and 'focus_desc' in cur_history,\
        'More focus objects than one, no focus objects!'
      focus_id = cur_history['focus_id']
      for attr in gvars.METAINFO['attributes']:
        if attr in cur_history['focus_desc']: break

      history = dialog['graph']['history'][:r_id + 1]
      for hist_id, hist_datum in enumerate(history):
        for obj in hist_datum['objects']:
          if obj['id'] == focus_id and attr in obj:
            dialog['graph']['history'][r_id + 1]['dependence'] = hist_id - 1
            break
  return dialog


def main(unused_argv):
  """Main method generates the CLEVR-Dialog dataset.
  """

  # Read the scene file.
  with open(FLAGS.scene_path, 'r') as file_id:
    scenes = json.load(file_id)

  # Read the synonyms file.
  with open(FLAGS.synonym_path, 'r') as file_id:
    synonyms = json.load(file_id)
  sorter = lambda x: len(x[0].split(' '))

  # Read the metainformation file.
  with open(FLAGS.metainfo_path, 'r') as file_id:
    gvars.METAINFO = json.load(file_id)
  tag_inv_map = {attr: tag for tag, attr in gvars.METAINFO['tag_map'].items()
                 if tag != '<P>'}
  gvars.METAINFO['tag_inv_map'] = tag_inv_map
  gvars.METAINFO['synonym_keys'] = sorted(synonyms.items(),
                                          key=sorter, reverse=True)

  # Add ids to objects.
  scenes = utils.add_object_ids(scenes)
  scenes = utils.clean_object_attributes(scenes)

  # Read the caption templates.
  template_paths = os.listdir(FLAGS.caption_template_root)
  cap_templates = []
  for ii in template_paths:
    with open(os.path.join(FLAGS.caption_template_root, ii), 'r') as file_id:
      cur_templates = json.load(file_id)
      cap_templates.extend(cur_templates)
  #utils.pretty_print_templates(cap_templates, 1)

  # Read the question templates.
  template_paths = os.listdir(FLAGS.question_template_root)
  ques_templates = []
  for ii in template_paths:
    with open(os.path.join(FLAGS.question_template_root, ii), 'r') as file_id:
     cur_templates = json.load(file_id)
     ques_templates.extend(cur_templates)
  #utils.pretty_print_templates(ques_templates, 1)

  # 1. Check if there a scene_id_file specified.
  # 2. Check if num_images is -1
  if FLAGS.scene_id_file != '':
    with open(FLAGS.scene_id_file, 'r') as file_id:
      missing_ids = [int(ii.strip('\n')) for ii in file_id.readlines()]
    print('Dialogs missing for scenes: %d' % len(missing_ids))

    # Create a image_index -> scenes list index dictionary
    image_list_id_dict = {ii['image_index']: index
                          for index, ii in enumerate(scenes['scenes'])}
    scenes_subset = [scenes['scenes'][image_list_id_dict[scene_id]]
                     for scene_id in missing_ids]

  elif FLAGS.num_images == -1:
    scenes_subset = scenes['scenes']

  else:
    scenes_subset = scenes['scenes'][0: FLAGS.num_images]

  # BFS for each scene.
  if FLAGS.num_workers == 1:
    # Single thread version.
    dialogs = []
    for index, scene in enumerate(scenes_subset):
      cur_time = time.strftime('%a-%d%b%y-%X', time.gmtime())
      print('Generating [ %s ] [ Worker: %d, Progress: %d/%d Scene:  %d ]' %\
            (cur_time, 0, index, len(scenes_subset), scene['image_index']))
      gen_dialog = generate_dialog_bfs(scene, cap_templates, ques_templates)
      dialogs.append(gen_dialog)

  else:
    # Multithread version.
    output_q = multiprocessing.Queue()
    jobs = []
    for worker_id in range(FLAGS.num_workers):
      allotment = scenes_subset[worker_id::FLAGS.num_workers]
      inputs = (allotment, cap_templates, ques_templates)
      inputs += (worker_id, output_q)

      process = multiprocessing.Process(target=worker, args=inputs)
      jobs.append(process)
      process.start()

    # Wait for all the jobs to finish and collect the output.
    final_results = {}
    for _ in jobs:
      final_results.update(output_q.get())
    for job in jobs:
      job.join()

    # Flatten and sort.
    final_results = [jj for _, ii in final_results.items() for jj in ii]
    dialogs = sorted(final_results, key=lambda x: x['image_index'])
  # utils.pretty_print_dialogs(dialogs)

  # Save the dialogs.
  print('Saving dialog at: %s' % FLAGS.save_path)
  with open(FLAGS.save_path, 'w') as file_id:
    json.dump(dialogs, file_id)


if __name__ == '__main__':
  gvars.initialize()
  app.run(main)
