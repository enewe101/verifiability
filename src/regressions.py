import matplotlib
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, metrics
from sklearn import svm
from sklearn.svm import SVC, SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, make_scorer, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV
import pickle

headers = ['sourceEntityPresence', 'sourcePlural', 'weaselWord', 'amodPresence', 'typeQuote', 'pronounSource', 'detSource', 'presumedVerb', 'sayVerb', 'beliefVerb', 'intendVerb', u'dets_a', u'dets_some', u'dets_the', u'amod_50-state', u'amod_common', u'amod_fellow', u'amod_female', u'amod_financial', u'amod_several', u'amod_top', u"sourceLemmaVocab_''", u'sourceLemmaVocab_Containers', u'sourceLemmaVocab_``', u'sourceLemmaVocab_advocate', u'sourceLemmaVocab_aide', u'sourceLemmaVocab_analyst', u'sourceLemmaVocab_announcer', u'sourceLemmaVocab_chairman', u'sourceLemmaVocab_company', u'sourceLemmaVocab_court', u'sourceLemmaVocab_datum', u'sourceLemmaVocab_dealer', u'sourceLemmaVocab_executive', u'sourceLemmaVocab_expert', u'sourceLemmaVocab_firm', u'sourceLemmaVocab_group', u'sourceLemmaVocab_he', u'sourceLemmaVocab_it', u'sourceLemmaVocab_japanese', u'sourceLemmaVocab_lawyer', u'sourceLemmaVocab_legislator', 'sourceLemmaVocab_loc', u'sourceLemmaVocab_manager', u'sourceLemmaVocab_market', u'sourceLemmaVocab_official', 'sourceLemmaVocab_org', u'sourceLemmaVocab_other', 'sourceLemmaVocab_pers', u'sourceLemmaVocab_planner', u'sourceLemmaVocab_researcher', u'sourceLemmaVocab_sign', u'sourceLemmaVocab_study', u'sourceLemmaVocab_survey', u'sourceLemmaVocab_teacher', u'sourceLemmaVocab_voice', u'sourceLemmaVocab_which', u'sourceLemmaVocab_who', u'sourceLemmaVocab_wisdom', u'sourceLemmaVocab_writer', u'typeEntities_DATE', u'typeEntities_LOCATION', u'typeEntities_MISC', u'typeEntities_NUMBER', u'typeEntities_ORGANIZATION', u'typeEntities_PERSON', u'weasel_some', u'cueBOW_accord', u'cueBOW_add', u'cueBOW_also', u'cueBOW_apologize', u'cueBOW_argue', u'cueBOW_ask', u'cueBOW_believe', u'cueBOW_call', u'cueBOW_characterize', u'cueBOW_charge', u'cueBOW_conclude', u'cueBOW_continue', u'cueBOW_deny', u'cueBOW_discipline', u'cueBOW_estimate', u'cueBOW_even', u'cueBOW_expect', u'cueBOW_explain', u'cueBOW_fear', u'cueBOW_find', u'cueBOW_indicate', u'cueBOW_insist', u'cueBOW_note', u'cueBOW_offer', u'cueBOW_order', 'cueBOW_org', u'cueBOW_publicly', u'cueBOW_purr', u'cueBOW_question', u'cueBOW_quip', u'cueBOW_quote', u'cueBOW_reply', u'cueBOW_report', u'cueBOW_say', u'cueBOW_seem', u'cueBOW_step', u'cueBOW_suggest', u'cueBOW_take', u'cueBOW_think', u'cueBOW_unusual', u'cueBOW_urge', u'cueBOW_view', u'cueBOW_want', u'cueBOW_worry', u"sourceBOW_'s", u'sourceBOW_50-state', u'sourceBOW_Containers', u'sourceBOW_Education', u'sourceBOW_Financial', u'sourceBOW_Friends', u'sourceBOW_Investor', u'sourceBOW_Mr.', u'sourceBOW_Mrs.', u'sourceBOW_N.M.', u'sourceBOW_Public', u'sourceBOW_School', u'sourceBOW_Sea', u'sourceBOW_September', u'sourceBOW_advocate', u'sourceBOW_aide', u'sourceBOW_also', u'sourceBOW_alumnus', u'sourceBOW_analyst', u'sourceBOW_announcer', u'sourceBOW_asian', u'sourceBOW_assistant', u'sourceBOW_association', u'sourceBOW_balanced', u'sourceBOW_bermuda-based', u'sourceBOW_case', u'sourceBOW_chairman', u'sourceBOW_common', u'sourceBOW_company', u'sourceBOW_concern', u'sourceBOW_court', u'sourceBOW_datum', u'sourceBOW_dealer', u'sourceBOW_deputy', u'sourceBOW_director', u'sourceBOW_east', u'sourceBOW_economy', u'sourceBOW_editor', u'sourceBOW_executive', u'sourceBOW_expert', u'sourceBOW_fellow', u'sourceBOW_female', u'sourceBOW_financial', u'sourceBOW_firm', u'sourceBOW_government', u'sourceBOW_group', u'sourceBOW_he', u'sourceBOW_health', u'sourceBOW_hotel', u'sourceBOW_human', u'sourceBOW_include', u'sourceBOW_it', u'sourceBOW_japanese', u'sourceBOW_lawyer', u'sourceBOW_legislator', u'sourceBOW_letter', 'sourceBOW_loc', u'sourceBOW_male', u'sourceBOW_manage', u'sourceBOW_manager', u'sourceBOW_market', u'sourceBOW_marketing', u'sourceBOW_media', u'sourceBOW_member', u'sourceBOW_minister', u'sourceBOW_office', u'sourceBOW_official', u'sourceBOW_one', u'sourceBOW_operate', 'sourceBOW_org', u'sourceBOW_organization', u'sourceBOW_over-the-counter', u'sourceBOW_partner', 'sourceBOW_pers', u'sourceBOW_planner', u'sourceBOW_political', u'sourceBOW_president', u'sourceBOW_prime', u'sourceBOW_prosecutor', u'sourceBOW_red-blooded', u'sourceBOW_release', u'sourceBOW_researcher', u'sourceBOW_resource', u'sourceBOW_retail', u'sourceBOW_safety', u'sourceBOW_sale', u'sourceBOW_school-research', u'sourceBOW_scientist', u'sourceBOW_secretary', u'sourceBOW_security', u'sourceBOW_self-regulatory', u'sourceBOW_several', u'sourceBOW_shipping', u'sourceBOW_sign', u'sourceBOW_specialist', u'sourceBOW_specialty', u'sourceBOW_steelmaker', u'sourceBOW_study', u'sourceBOW_survey', u'sourceBOW_teacher', u'sourceBOW_top', u'sourceBOW_trade', u'sourceBOW_vice', u'sourceBOW_voice', u'sourceBOW_which', u'sourceBOW_who', u'sourceBOW_wisdom', u'sourceBOW_writer', u"contentBOW_'40", u"contentBOW_'50s", u"contentBOW_'s", u'contentBOW_--', u'contentBOW_1', u'contentBOW_1.1', u'contentBOW_1.82', u'contentBOW_10', u'contentBOW_100', u'contentBOW_11', u'contentBOW_11.6', u'contentBOW_14', u'contentBOW_1990', u'contentBOW_2', u'contentBOW_20', u'contentBOW_300', u'contentBOW_3057', u'contentBOW_35', u'contentBOW_36-day', u'contentBOW_4,000', u'contentBOW_400', u'contentBOW_42.5', u'contentBOW_445', u'contentBOW_47.5', u'contentBOW_50', u'contentBOW_68', u'contentBOW_70', u'contentBOW_75', u'contentBOW_8', u'contentBOW_84.29', u'contentBOW_9', u'contentBOW_90', u'contentBOW_AIDS', u'contentBOW_December', u'contentBOW_Fund', u'contentBOW_I', u'contentBOW_January', u'contentBOW_March', u'contentBOW_Monday', u'contentBOW_Mr.', u'contentBOW_Mrs.', u'contentBOW_Nov.', u'contentBOW_October', u'contentBOW_Rally', u'contentBOW_September', u'contentBOW_Series', u'contentBOW_Thursday', u'contentBOW_Tuesday', u'contentBOW_able', u'contentBOW_accept', u'contentBOW_access', u'contentBOW_account', u'contentBOW_achievement', u'contentBOW_acquire', u'contentBOW_acquisition', u'contentBOW_across', u'contentBOW_act', u'contentBOW_action', u'contentBOW_actively', u'contentBOW_activity', u'contentBOW_ad', u'contentBOW_administration', u'contentBOW_advertising', u'contentBOW_aerospace', u'contentBOW_affiliate', u'contentBOW_age', u'contentBOW_ago', u'contentBOW_agree', u'contentBOW_allege', u'contentBOW_almost', u'contentBOW_already', u'contentBOW_although', u'contentBOW_american', u'contentBOW_annual', u'contentBOW_anything', u'contentBOW_apiece', u'contentBOW_approach', u'contentBOW_around', u'contentBOW_asbestos-related', u'contentBOW_asian', u'contentBOW_asset', u'contentBOW_assistant', u'contentBOW_attention', u'contentBOW_audit', u'contentBOW_available', u'contentBOW_avoid', u'contentBOW_back', u'contentBOW_balance', u'contentBOW_ban', u'contentBOW_bank', u'contentBOW_bankruptcy', u'contentBOW_bargain', u'contentBOW_base', u'contentBOW_bearing', u'contentBOW_become', u'contentBOW_begin', u'contentBOW_bell', u'contentBOW_beneficiary', u'contentBOW_bid', u'contentBOW_bidding', u'contentBOW_big', u'contentBOW_bill', u'contentBOW_billion', u'contentBOW_black', u'contentBOW_blip', u'contentBOW_board', u'contentBOW_bottle', u'contentBOW_bring', u'contentBOW_build', u'contentBOW_bullet', u'contentBOW_buy', u'contentBOW_campaign', u'contentBOW_campaigner', u'contentBOW_campaigning', u'contentBOW_candidate', u'contentBOW_capital', u'contentBOW_capitalist', u'contentBOW_car-safety', u'contentBOW_care', u'contentBOW_carry', u'contentBOW_case', u'contentBOW_cash', u'contentBOW_catch', u'contentBOW_certain', u'contentBOW_cheating', u'contentBOW_child', u'contentBOW_chinese', u'contentBOW_choose', u'contentBOW_claim', u'contentBOW_client', u'contentBOW_cocky', u'contentBOW_come', u'contentBOW_comfortable', u'contentBOW_comment', u'contentBOW_common', u'contentBOW_company', u'contentBOW_compare', u'contentBOW_complete', u'contentBOW_conditional', u'contentBOW_confuse', u'contentBOW_construction', u'contentBOW_consumer', u'contentBOW_contact', u'contentBOW_content', u'contentBOW_continue', u'contentBOW_contract', u'contentBOW_convertible', u'contentBOW_cooperation', u'contentBOW_cop-killer', u'contentBOW_corporate', u'contentBOW_corporation', u'contentBOW_correct', u'contentBOW_could', u'contentBOW_country', u'contentBOW_court', u'contentBOW_craft', u'contentBOW_credit', u'contentBOW_currency', u'contentBOW_current', u'contentBOW_dam', u'contentBOW_damage', u'contentBOW_datum', u'contentBOW_declare', u'contentBOW_decline', u'contentBOW_decrease', u'contentBOW_default', u'contentBOW_defeat', u'contentBOW_demand', u'contentBOW_demonstration', u'contentBOW_department', u'contentBOW_derivative', u'contentBOW_design', u'contentBOW_device', u'contentBOW_differ', u'contentBOW_diminish', u'contentBOW_direction', u'contentBOW_disciplinary', u'contentBOW_disclosure', u'contentBOW_discover', u'contentBOW_disease', u'contentBOW_disproportionate', u'contentBOW_diversify', u'contentBOW_dividend', u'contentBOW_dollar', u'contentBOW_domestic', u'contentBOW_double', u'contentBOW_dramatic', u'contentBOW_drawing', u'contentBOW_eager', u'contentBOW_early', u'contentBOW_easier', u'contentBOW_economy', u'contentBOW_educator', u'contentBOW_effective', u'contentBOW_effort', u'contentBOW_elect', u'contentBOW_emerge', u'contentBOW_empty', u'contentBOW_end', u'contentBOW_energy', u'contentBOW_engage', u'contentBOW_enough', u'contentBOW_entrench', u'contentBOW_environment', u'contentBOW_environmental', u'contentBOW_era', u'contentBOW_espouse', u'contentBOW_even', u'contentBOW_event', u'contentBOW_ever', u'contentBOW_exclusive', u'contentBOW_exercise', u'contentBOW_expense', u'contentBOW_export', u'contentBOW_exposure', u'contentBOW_extend', u'contentBOW_extent', u'contentBOW_fall', u'contentBOW_fatten', u'contentBOW_federal', u'contentBOW_federally', u'contentBOW_feel', u'contentBOW_fetal-tissue', u'contentBOW_field', u'contentBOW_file', u'contentBOW_finance', u'contentBOW_firm', u'contentBOW_first', u'contentBOW_five', u'contentBOW_fixed-rate', u'contentBOW_flap', u'contentBOW_flightiness', u'contentBOW_flow', u'contentBOW_food-shop', u'contentBOW_force', u'contentBOW_foreign', u'contentBOW_foreign-stock', u'contentBOW_four', u'contentBOW_fraud', u'contentBOW_frozen', u'contentBOW_fund', u'contentBOW_funding', u'contentBOW_get', u'contentBOW_global', u'contentBOW_go', u'contentBOW_gold', u'contentBOW_good', u'contentBOW_government', u'contentBOW_governor', u'contentBOW_grant', u'contentBOW_greatly', u'contentBOW_group', u'contentBOW_grow', u'contentBOW_growth', u'contentBOW_guild', u'contentBOW_half', u'contentBOW_halt', u'contentBOW_hand', u'contentBOW_hazardous', u'contentBOW_he', u'contentBOW_hearing', u'contentBOW_hefty', u'contentBOW_help', u'contentBOW_high', u'contentBOW_higher', u'contentBOW_highly', u'contentBOW_hold', u'contentBOW_home', u'contentBOW_hope', u'contentBOW_hour', u'contentBOW_hurt', u'contentBOW_idea', u'contentBOW_identify', u'contentBOW_ideological', u'contentBOW_important', u'contentBOW_impose', u'contentBOW_increase', u'contentBOW_increasingly', u'contentBOW_indefinitely', u'contentBOW_indicate', u'contentBOW_individual', u'contentBOW_industry', u'contentBOW_inflated', u'contentBOW_information', u'contentBOW_inkling', u'contentBOW_integration', u'contentBOW_interest', u'contentBOW_interested', u'contentBOW_international', u'contentBOW_investigate', u'contentBOW_investment', u'contentBOW_investor', u'contentBOW_involve', u'contentBOW_issue', u'contentBOW_it', u'contentBOW_its', u'contentBOW_japanese', u'contentBOW_judge', u'contentBOW_keep', u'contentBOW_kind', u'contentBOW_know', u'contentBOW_knowledge', u'contentBOW_lack', u'contentBOW_language', u'contentBOW_large', u'contentBOW_largest', u'contentBOW_later', u'contentBOW_law', u'contentBOW_less', u'contentBOW_level', u'contentBOW_license', u'contentBOW_life', u'contentBOW_light', u'contentBOW_likely', u'contentBOW_limit', u'contentBOW_list', u'contentBOW_little', 'contentBOW_loc', u'contentBOW_local', u'contentBOW_lock', u'contentBOW_look', u'contentBOW_lucky', u'contentBOW_magazine', u'contentBOW_mail', u'contentBOW_main', u'contentBOW_mainland', u'contentBOW_major', u'contentBOW_majority', u'contentBOW_make', u'contentBOW_maker', u'contentBOW_manage', u'contentBOW_management', u'contentBOW_manager', u'contentBOW_manufacturing', u'contentBOW_many', u'contentBOW_marble', u'contentBOW_mark', u'contentBOW_markdown', u'contentBOW_market', u'contentBOW_marketer', u'contentBOW_marketing', u'contentBOW_markup', u'contentBOW_material', u'contentBOW_materialistic', u'contentBOW_may', u'contentBOW_mayor', u'contentBOW_meaning', u'contentBOW_meeting', u'contentBOW_member', u'contentBOW_merger', u'contentBOW_million', u'contentBOW_mistake', u'contentBOW_modestly', u'contentBOW_money', u'contentBOW_month', u'contentBOW_morale-damaging', u'contentBOW_motive', u'contentBOW_move', u'contentBOW_much', u'contentBOW_mudslinging', u'contentBOW_mushy', u'contentBOW_narrow', u'contentBOW_nation', u'contentBOW_national', u'contentBOW_nearly', u'contentBOW_need', u'contentBOW_negative', u'contentBOW_neighbor', u'contentBOW_never', u'contentBOW_nevertheless', u'contentBOW_new', u'contentBOW_newer', u'contentBOW_newly', u'contentBOW_next', u'contentBOW_nine-member', u'contentBOW_noodle', u'contentBOW_number', u'contentBOW_offer', u'contentBOW_office', u'contentBOW_offset', u'contentBOW_often', u'contentBOW_old', u'contentBOW_on-campus', u'contentBOW_one', u'contentBOW_one-third', u'contentBOW_only', u'contentBOW_onus', u'contentBOW_operating', u'contentBOW_operation', u'contentBOW_opportunity', u'contentBOW_oppose', u'contentBOW_order', 'contentBOW_org', u'contentBOW_organization', u'contentBOW_original', u'contentBOW_outright', u'contentBOW_outside', u'contentBOW_parallel', u'contentBOW_part', u'contentBOW_partner', u'contentBOW_past', u'contentBOW_pasta', u'contentBOW_patent', u'contentBOW_pay', u'contentBOW_payout', u'contentBOW_peal', u'contentBOW_people', u'contentBOW_perhaps', u'contentBOW_period', 'contentBOW_pers', u'contentBOW_pick', u'contentBOW_place', u'contentBOW_plan', u'contentBOW_plant', u'contentBOW_policy', u'contentBOW_poor', u'contentBOW_popular', u'contentBOW_possible', u'contentBOW_post', u'contentBOW_potential', u'contentBOW_power', u'contentBOW_pre-cooked', u'contentBOW_preferred', u'contentBOW_premium', u'contentBOW_president', u'contentBOW_pretty', u'contentBOW_prevent', u'contentBOW_price', u'contentBOW_pride', u'contentBOW_priority', u'contentBOW_privately', u'contentBOW_proceeds', u'contentBOW_process', u'contentBOW_produce', u'contentBOW_product', u'contentBOW_profit', u'contentBOW_program', u'contentBOW_program-trading', u'contentBOW_programming', u'contentBOW_progress', u'contentBOW_project', u'contentBOW_prominently', u'contentBOW_property', u'contentBOW_proposal', u'contentBOW_prosecutor', u'contentBOW_provide', u'contentBOW_public', u'contentBOW_purchasing', u'contentBOW_put', u'contentBOW_quarter', u'contentBOW_queer', u'contentBOW_question', u'contentBOW_quickly', u'contentBOW_quite', u'contentBOW_raise', u'contentBOW_range', u'contentBOW_rank', u'contentBOW_rate', u'contentBOW_rather', u'contentBOW_recede', u'contentBOW_recent', u'contentBOW_recognition', u'contentBOW_recognize', u'contentBOW_refer', u'contentBOW_reflect', u'contentBOW_refund', u'contentBOW_reject', u'contentBOW_relation', u'contentBOW_relatively', u'contentBOW_remain', u'contentBOW_remove', u'contentBOW_reorganization', u'contentBOW_report', u'contentBOW_require', u'contentBOW_requirement', u'contentBOW_research', u'contentBOW_reserve', u'contentBOW_resistant', u'contentBOW_result', u'contentBOW_return', u'contentBOW_revenue', u'contentBOW_right', u'contentBOW_rights', u'contentBOW_ringer', u'contentBOW_rise', u'contentBOW_risk', u'contentBOW_robustly', u'contentBOW_role', u'contentBOW_rule', u'contentBOW_run', u'contentBOW_sale', u'contentBOW_say', u'contentBOW_scenario', u'contentBOW_schedule', u'contentBOW_scientific', u'contentBOW_score', u'contentBOW_seat', u'contentBOW_sector', u'contentBOW_security', u'contentBOW_seduce', u'contentBOW_see', u'contentBOW_seek', u'contentBOW_sell', u'contentBOW_settle', u'contentBOW_several', u'contentBOW_sex', u'contentBOW_share', u'contentBOW_shareholder', u'contentBOW_she', u'contentBOW_shipment', u'contentBOW_short-term', u'contentBOW_shut', u'contentBOW_side', u'contentBOW_significant', u'contentBOW_since', u'contentBOW_single-digit', u'contentBOW_smattering', u'contentBOW_smoking', u'contentBOW_socialist', u'contentBOW_sort', u'contentBOW_special', u'contentBOW_specifics', u'contentBOW_spender', u'contentBOW_sphere', u'contentBOW_spring', u'contentBOW_spur', u'contentBOW_stake', u'contentBOW_stalemate', u'contentBOW_standardized', u'contentBOW_start', u'contentBOW_state', u'contentBOW_steel', u'contentBOW_stigma', u'contentBOW_still', u'contentBOW_stock', u'contentBOW_stop', u'contentBOW_strength', u'contentBOW_strike', u'contentBOW_strong', u'contentBOW_student', u'contentBOW_stupid', u'contentBOW_style', u'contentBOW_sub-segment', u'contentBOW_subcommittee', u'contentBOW_subsidiary', u'contentBOW_sue', u'contentBOW_suffer', u'contentBOW_superior', u'contentBOW_supplier', u'contentBOW_supply', u'contentBOW_support', u'contentBOW_survey', u'contentBOW_swap', u'contentBOW_swing', u'contentBOW_tailor', u'contentBOW_take', u'contentBOW_talk', u'contentBOW_target', u'contentBOW_tea', u'contentBOW_teach', u'contentBOW_teacher', u'contentBOW_telecommunications', u'contentBOW_television', u'contentBOW_tell', u'contentBOW_test', u'contentBOW_they', u'contentBOW_three', u'contentBOW_time', u'contentBOW_tired', u'contentBOW_tissue-transplant', u'contentBOW_today', u'contentBOW_top', u'contentBOW_trade', u'contentBOW_trading', u'contentBOW_traditional', u'contentBOW_transplant', u'contentBOW_trial', u'contentBOW_truck', u'contentBOW_turn', u'contentBOW_tv', u'contentBOW_twin', u'contentBOW_two', u'contentBOW_u.s.-japan', u'contentBOW_unabated', u'contentBOW_underlie', u'contentBOW_underprivileged', u'contentBOW_unfair', u'contentBOW_unit', u'contentBOW_unlike', u'contentBOW_unreasonable', u'contentBOW_use', u'contentBOW_usher', u'contentBOW_utility', u'contentBOW_vague', u'contentBOW_valuation', u'contentBOW_value', u'contentBOW_van', u'contentBOW_various', u'contentBOW_vehicle', u'contentBOW_victim', u'contentBOW_viewpoint', u'contentBOW_violation', u'contentBOW_volatile', u'contentBOW_volume', u'contentBOW_want', u'contentBOW_waste', u'contentBOW_we', u'contentBOW_weakening', u'contentBOW_week', u'contentBOW_well', u'contentBOW_western', u'contentBOW_what', u'contentBOW_whether', u'contentBOW_which', u'contentBOW_widow', u'contentBOW_win', u'contentBOW_wine', u'contentBOW_within', u'contentBOW_without', u'contentBOW_woman', u'contentBOW_work', u'contentBOW_worker', u'contentBOW_worst-case', u'contentBOW_would', u'contentBOW_write', u'contentBOW_year', u'contentBOW_yen', u'contentBOW_yield', u'contentBOW_you', u'contentBOW_young', u'contentBOW_youth']


def readData(datafile):
	data = pickle.load(open(datafile, 'rb'))

	df = np.asarray(data)
	df = df.astype(np.float)

	Y_values = df[:, 0]
	Y_values = Y_values * 100
	X_Values = df[:,1:]

	X_train, X_test, y_train, y_test = train_test_split(X_Values, Y_values, test_size=0.1, random_state=200)
	return X_train, X_test, y_train, y_test

def bin_data(y_train, y_test):
	bins = np.array([10,20,30,40,50,60,70,80,90,100])

	binned_y_train = np.digitize(y_train, bins, right=False)
	binned_y_test = np.digitize(y_test, bins, right=False)

	return binned_y_train, binned_y_test

#####################   linear model   #####################

def linModel(X_train, y_train):
	linModel = linear_model.LinearRegression()
	linModel.fit(X_train, y_train)
	return linModel

######################   Lasso Regularization    #####################	

def lasso(X_train, y_train):
	lasso = linear_model.LassoCV(cv = 5)
	lasso.fit(X_train, y_train)
	return lasso

######################   Hyperparameter Tuning SVR    #####################	

def run_svr(X_train, y_train):

	tuned_parameters = [
		{'C': [1, 10, 100, 1000, 10000], 'kernel': ['linear']},
		{'C': [1, 10, 100, 1000, 10000], 'gamma': ['auto', 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']},
		{'C': [1, 10, 100, 1000, 10000], 'coef0': [0, 0.5, 1, 1.5, 2],'gamma': ['auto', 0.1, 0.01, 0.001, 0.0001], 'degree': [2,3,4,5,6], 'kernel': ['poly']},
		{'C': [1, 10, 100, 1000, 10000], 'gamma': ['auto', 0.1, 0.01, 0.001, 0.0001], 'coef0': [0, 0.5, 1, 1.5, 2],  'kernel': ['sigmoid']},
	]

	clf = GridSearchCV(SVR(C=1), tuned_parameters, cv=5, scoring='neg_mean_squared_error')
	clf.fit(X_train, y_train)
	print(clf.best_params_)
	'''
	print("Grid scores on development set:")
	print
	means = clf.cv_results_['mean_test_score']
	stds = clf.cv_results_['std_test_score']
	for mean, std, params in zip(means, stds, clf.cv_results_['params']):
		print("%0.3f (+/-%0.03f) for %r"
			% (mean, std * 2, params))
	print
	'''

	return clf.best_estimator_

######################   Hyperparameter Tuning SVC    #####################	

def run_svc(X_train, y_train):

	tuned_parameters = [
		{'C': [1, 10, 100, 1000, 10000], 'kernel': ['linear']},
		{'C': [1, 10, 100, 1000, 10000], 'gamma': ['auto', 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']},
		{'C': [1, 10, 100, 1000, 10000], 'coef0': [0, 0.5, 1, 1.5, 2],'gamma': ['auto', 0.1, 0.01, 0.001, 0.0001], 'degree': [2,3,4,5,6], 'kernel': ['poly']},
		{'C': [1, 10, 100, 1000, 10000], 'gamma': ['auto', 0.1, 0.01, 0.001, 0.0001], 'coef0': [0, 0.5, 1, 1.5, 2],  'kernel': ['sigmoid']},
	]

	clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring='neg_mean_squared_error')
	clf.fit(X_train, y_train)
	print(clf.best_params_)
	'''
	print("Grid scores on development set:")
	print
	means = clf.cv_results_['mean_test_score']
	stds = clf.cv_results_['std_test_score']
	for mean, std, params in zip(means, stds, clf.cv_results_['params']):
		print("%0.3f (+/-%0.03f) for %r"
			% (mean, std * 2, params))
	print
	'''

	return clf.best_estimator_

######################   CREATING BINNING SCORERS : per 5   #####################	
def myBinMeanSquaredError(real, predicted):
	bins = np.array([5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100])

	realBin = np.digitize(real, bins, right=False)
	predBin = np.digitize(predicted, bins, right=False)
	return mean_squared_error(realBin, predBin)

def myBinMeanAbsoluteError(real, predicted):
	bins = np.array([5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100])

	realBin = np.digitize(real, bins, right=False)
	predBin = np.digitize(predicted, bins, right=False)
	return mean_absolute_error(realBin, predBin)

def myBinAccuracy(real, predicted):
	bins = np.array([5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100])

	realBin = np.digitize(real, bins, right=False)
	predBin = np.digitize(predicted, bins, right=False)

	return accuracy_score(realBin, predBin)


######################   CREATING BINNING SCORERS : per 10   #####################	
def myBinMeanSquaredErrorDecile(real, predicted):
	bins = np.array([10,20,30,40,50,60,70,80,90,100])

	realBin = np.digitize(real, bins, right=False)
	predBin = np.digitize(predicted, bins, right=False)
	return mean_squared_error(realBin, predBin)

def myBinMeanAbsoluteErrorDecile(real, predicted):
	bins = np.array([10,20,30,40,50,60,70,80,90,100])

	realBin = np.digitize(real, bins, right=False)
	predBin = np.digitize(predicted, bins, right=False)
	return mean_absolute_error(realBin, predBin)

def myBinAccuracyDecile(real, predicted):
	bins = np.array([10,20,30,40,50,60,70,80,90,100])

	realBin = np.digitize(real, bins, right=False)
	predBin = np.digitize(predicted, bins, right=False)

	return accuracy_score(realBin, predBin)


######################   QUANTITATIVE ERROR ANALYSIS    #####################	
def errorAnal(model,X_train, y_train, X_test, y_test):

	MSE = abs(np.mean(cross_val_score(model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')))
	MAE = abs(np.mean(cross_val_score(model, X_train, y_train, cv=10, scoring='neg_mean_absolute_error')))

	print("Mean squared error: %.2f" % MSE)
	print("Root Mean squared error: %.2f" % math.sqrt(MSE))
	print("Mean Absolute Error: %.2f" % MAE)


	MSE_bins_scorer = make_scorer(myBinMeanSquaredError, greater_is_better=False)
	MSEbinned = abs(np.mean(cross_val_score(model, X_train, y_train, cv=10, scoring=MSE_bins_scorer)))
	
	MAE_bins_scorer = make_scorer(myBinMeanAbsoluteError, greater_is_better=False)
	MAEbinned = abs(np.mean(cross_val_score(model, X_train, y_train, cv=10, scoring=MAE_bins_scorer)))

	accuracy_bins_scorer = make_scorer(myBinAccuracy, greater_is_better=True)
	accuracy_binned = abs(np.mean(cross_val_score(model, X_train, y_train, cv=10, scoring=accuracy_bins_scorer)))

	print("Binned (5): Mean squared error: %.2f" % MSEbinned)
	print("Binned (5): Root Mean squared error: %.2f" % math.sqrt(MSEbinned))
	print("Binned (5): Mean Absolute Error: %.2f" % MAEbinned)
	print("Binned (5): Accuracy: %.2f" % accuracy_binned)

	MSE_decile_bins_scorer = make_scorer(myBinMeanSquaredErrorDecile, greater_is_better=False)
	MSE_decile_binned = abs(np.mean(cross_val_score(model, X_train, y_train, cv=10, scoring=MSE_decile_bins_scorer)))
	
	MAE_decile_bins_scorer = make_scorer(myBinMeanAbsoluteErrorDecile, greater_is_better=False)
	MAE_decile_binned = abs(np.mean(cross_val_score(model, X_train, y_train, cv=10, scoring=MAE_decile_bins_scorer)))

	accuracy_decile_bins_scorer = make_scorer(myBinAccuracyDecile, greater_is_better=True)
	accuracy_decile_binned = abs(np.mean(cross_val_score(model, X_train, y_train, cv=10, scoring=accuracy_decile_bins_scorer)))

	print("Binned (10): Mean squared error: %.2f" % MSE_decile_binned)
	print("Binned (10): Root Mean squared error: %.2f" % math.sqrt(MSE_decile_binned))
	print("Binned (10): Mean Absolute Error: %.2f" % MAE_decile_binned)
	print("Binned (10): Accuracy: %.2f" % accuracy_decile_binned)

	predictions = model.predict(X_test)
	print "  Prediction  |  Real   |  Squared Error"
	for prediction in enumerate(predictions):
		index = prediction[0]
		print prediction[1], y_test[index], ((prediction[1] - y_test[index]) **2 )

def binnedErrorAnalysis(model,X_train, y_train, X_test, y_test):
	MSE = abs(np.mean(cross_val_score(model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')))
	MAE = abs(np.mean(cross_val_score(model, X_train, y_train, cv=10, scoring='neg_mean_absolute_error')))

	print("Mean squared error: %.2f" % MSE)
	print("Root Mean squared error: %.2f" % math.sqrt(MSE))
	print("Mean Absolute Error: %.2f" % MAE)

	print
	print "classification report on training set"
	y_pred = cross_val_predict(model, X_train, y_train, cv=10)
	print(classification_report(y_train, y_pred))

	print
	print "classification report on test set"
	y_pred_test = model.predict(X_test)
	print(classification_report(y_test, y_pred_test))

	predictions = model.predict(X_test)
	for prediction in enumerate(predictions):
		index = prediction[0]
		print prediction[1], y_test[index], ((prediction[1] - y_test[index]) **2 )


def ablationTesting(model, X_train, y_train):
	scores = []

	for i in range(X_train.shape[1]):
		score = cross_val_score(model, X_train[:, i:i+1], y_train, scoring="r2", cv=5)
		scores.append((round(np.mean(score), 3), headers[i]))
	
	sortedScores = sorted(scores, reverse=True)
	print sortedScores[:10]

	score_numbers, feat_name = zip(*sortedScores)

def main():

	print "Reading Data"
	X_train, X_test, y_train, y_test = readData('data/verifiabilityNumFeatures')
	print
	print
	
	print "----Linear Regression----"
	linRegression = linModel(X_train, y_train)
	print
	print "Error Analysis"
	print
	errorAnal(linRegression, X_train, y_train, X_test, y_test)
	print
	print "Ablation Testing"
	print
	ablationTesting(linRegression,X_train, y_train)
	print
	print

	print "----Linear Regression with Lasso Regularization----"
	lassoModel = lasso(X_train, y_train)
	print
	print "Error Analysis"
	errorAnal(lassoModel, X_train, y_train, X_test, y_test)
	print
	print "Ablation Testing"
	ablationTesting(lassoModel,X_train, y_train)
	print
	print

	print "----Support Vector Regression: Training + Hyper Parameter Tuning----"
	best_svr = run_svr(X_train, y_train)
	print
	print "Error Analysis"
	errorAnal(best_svr, X_train, y_train, X_test, y_test)
	print
	print "Ablation Testing"
	ablationTesting(best_svr,X_train, y_train)
	print
	print
	

	y_train_binned, y_test_binned = bin_data(y_train, y_test)

	print "----Multiclass Support Vector Machine: Training + Hyper Parameter Tuning----"
	best_svc = run_svc(X_train, y_train_binned)
	print
	print "Error Analysis"
	binnedErrorAnalysis(best_svc, X_train, y_train_binned, X_test, y_test_binned)
	print
	print "Ablation Testing"
	ablationTesting(best_svc,X_train, y_train_binned)
	print
	print


if __name__ == '__main__':
   main()

