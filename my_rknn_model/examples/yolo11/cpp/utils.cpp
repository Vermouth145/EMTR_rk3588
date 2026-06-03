#include "BYTETracker.h"
#include <unordered_map>
#include <unordered_set>
#include "lapjv.h"

vector<STrack*> BYTETracker::joint_stracks(vector<STrack*> &tlista, vector<STrack> &tlistb)
{
	std::unordered_set<int> exists;
	vector<STrack*> res;
	res.reserve(tlista.size() + tlistb.size());
	for (int i = 0; i < (int)tlista.size(); i++)
	{
		exists.insert(tlista[i]->track_id);
		res.push_back(tlista[i]);
	}
	for (int i = 0; i < (int)tlistb.size(); i++)
	{
		int tid = tlistb[i].track_id;
		if (exists.find(tid) == exists.end())
		{
			exists.insert(tid);
			res.push_back(&tlistb[i]);
		}
	}
	return res;
}

vector<STrack> BYTETracker::joint_stracks(vector<STrack> &tlista, vector<STrack> &tlistb)
{
	std::unordered_set<int> exists;
	vector<STrack> res;
	res.reserve(tlista.size() + tlistb.size());
	for (int i = 0; i < (int)tlista.size(); i++)
	{
		exists.insert(tlista[i].track_id);
		res.push_back(tlista[i]);
	}
	for (int i = 0; i < (int)tlistb.size(); i++)
	{
		int tid = tlistb[i].track_id;
		if (exists.find(tid) == exists.end())
		{
			exists.insert(tid);
			res.push_back(tlistb[i]);
		}
	}
	return res;
}

vector<STrack> BYTETracker::sub_stracks(vector<STrack> &tlista, vector<STrack> &tlistb)
{
	std::unordered_map<int, int> track_index;
	track_index.reserve(tlista.size());
	for (int i = 0; i < (int)tlista.size(); i++)
	{
		track_index[tlista[i].track_id] = i;
	}
	for (int i = 0; i < (int)tlistb.size(); i++)
	{
		track_index.erase(tlistb[i].track_id);
	}

	vector<STrack> res;
	res.reserve(track_index.size());
	for (const auto &kv : track_index)
	{
		res.push_back(tlista[kv.second]);
	}
	return res;
}

void BYTETracker::remove_duplicate_stracks(vector<STrack> &resa, vector<STrack> &resb, vector<STrack> &stracksa, vector<STrack> &stracksb)
{
	vector<vector<float> > pdist = iou_distance(stracksa, stracksb);
	vector<pair<int, int> > pairs;
	for (int i = 0; i < (int)pdist.size(); i++)
	{
		for (int j = 0; j < (int)pdist[i].size(); j++)
		{
			if (pdist[i][j] < 0.15)
			{
				pairs.push_back(pair<int, int>(i, j));
			}
		}
	}

	std::unordered_set<int> dupa, dupb;
	for (int i = 0; i < (int)pairs.size(); i++)
	{
		int timep = stracksa[pairs[i].first].frame_id - stracksa[pairs[i].first].start_frame;
		int timeq = stracksb[pairs[i].second].frame_id - stracksb[pairs[i].second].start_frame;
		if (timep > timeq)
			dupb.insert(pairs[i].second);
		else
			dupa.insert(pairs[i].first);
	}

	resa.reserve(stracksa.size());
	for (int i = 0; i < (int)stracksa.size(); i++)
	{
		if (dupa.find(i) == dupa.end())
		{
			resa.push_back(stracksa[i]);
		}
	}

	resb.reserve(stracksb.size());
	for (int i = 0; i < (int)stracksb.size(); i++)
	{
		if (dupb.find(i) == dupb.end())
		{
			resb.push_back(stracksb[i]);
		}
	}
}

void BYTETracker::linear_assignment(vector<vector<float> > &cost_matrix, int cost_matrix_size, int cost_matrix_size_size, float thresh,
	vector<vector<int> > &matches, vector<int> &unmatched_a, vector<int> &unmatched_b)
{
	if (cost_matrix.size() == 0)
	{
		for (int i = 0; i < cost_matrix_size; i++)
		{
			unmatched_a.push_back(i);
		}
		for (int i = 0; i < cost_matrix_size_size; i++)
		{
			unmatched_b.push_back(i);
		}
		return;
	}

	vector<int> rowsol; vector<int> colsol;
	float c = lapjv(cost_matrix, rowsol, colsol, true, thresh);
	for (int i = 0; i < rowsol.size(); i++)
	{
		if (rowsol[i] >= 0)
		{
			vector<int> match;
			match.push_back(i);
			match.push_back(rowsol[i]);
			matches.push_back(match);
		}
		else
		{
			unmatched_a.push_back(i);
		}
	}

	for (int i = 0; i < colsol.size(); i++)
	{
		if (colsol[i] < 0)
		{
			unmatched_b.push_back(i);
		}
	}
}

vector<vector<float> > BYTETracker::ious(vector<vector<float> > &atlbrs, vector<vector<float> > &btlbrs)
{
	vector<vector<float> > ious;
	if (atlbrs.size()*btlbrs.size() == 0)
		return ious;

	ious.resize(atlbrs.size());
	for (int i = 0; i < ious.size(); i++)
	{
		ious[i].resize(btlbrs.size());
	}

	//bbox_ious
	for (int k = 0; k < btlbrs.size(); k++)
	{
		vector<float> ious_tmp;
		float box_area = (btlbrs[k][2] - btlbrs[k][0] + 1)*(btlbrs[k][3] - btlbrs[k][1] + 1);
		for (int n = 0; n < atlbrs.size(); n++)
		{
			float iw = min(atlbrs[n][2], btlbrs[k][2]) - max(atlbrs[n][0], btlbrs[k][0]) + 1;
			if (iw > 0)
			{
				float ih = min(atlbrs[n][3], btlbrs[k][3]) - max(atlbrs[n][1], btlbrs[k][1]) + 1;
				if(ih > 0)
				{
					float ua = (atlbrs[n][2] - atlbrs[n][0] + 1)*(atlbrs[n][3] - atlbrs[n][1] + 1) + box_area - iw * ih;
					ious[n][k] = iw * ih / ua;
				}
				else
				{
					ious[n][k] = 0.0;
				}
			}
			else
			{
				ious[n][k] = 0.0;
			}
		}
	}

	return ious;
}

vector<vector<float> > BYTETracker::iou_distance(vector<STrack*> &atracks, vector<STrack> &btracks, int &dist_size, int &dist_size_size)
{
	vector<vector<float> > cost_matrix;
	if (atracks.size() * btracks.size() == 0)
	{
		dist_size = atracks.size();
		dist_size_size = btracks.size();
		return cost_matrix;
	}
	vector<vector<float> > atlbrs, btlbrs;
	for (int i = 0; i < atracks.size(); i++)
	{
		atlbrs.push_back(atracks[i]->tlbr);
	}
	for (int i = 0; i < btracks.size(); i++)
	{
		btlbrs.push_back(btracks[i].tlbr);
	}

	dist_size = atracks.size();
	dist_size_size = btracks.size();

	vector<vector<float> > _ious = ious(atlbrs, btlbrs);
	
	for (int i = 0; i < _ious.size();i++)
	{
		vector<float> _iou;
		for (int j = 0; j < _ious[i].size(); j++)
		{
			_iou.push_back(1 - _ious[i][j]);
		}
		cost_matrix.push_back(_iou);
	}

	return cost_matrix;
}

vector<vector<float> > BYTETracker::iou_distance(vector<STrack> &atracks, vector<STrack> &btracks)
{
	vector<vector<float> > atlbrs, btlbrs;
	for (int i = 0; i < atracks.size(); i++)
	{
		atlbrs.push_back(atracks[i].tlbr);
	}
	for (int i = 0; i < btracks.size(); i++)
	{
		btlbrs.push_back(btracks[i].tlbr);
	}

	vector<vector<float> > _ious = ious(atlbrs, btlbrs);
	vector<vector<float> > cost_matrix;
	for (int i = 0; i < _ious.size(); i++)
	{
		vector<float> _iou;
		for (int j = 0; j < _ious[i].size(); j++)
		{
			_iou.push_back(1 - _ious[i][j]);
		}
		cost_matrix.push_back(_iou);
	}

	return cost_matrix;
}

double BYTETracker::lapjv(const vector<vector<float> > &cost, vector<int> &rowsol, vector<int> &colsol,
	bool extend_cost, float cost_limit, bool return_cost)
{
	vector<vector<float> > cost_c;
	cost_c.assign(cost.begin(), cost.end());

	vector<vector<float> > cost_c_extended;

	int n_rows = cost.size();
	int n_cols = cost[0].size();
	rowsol.resize(n_rows);
	colsol.resize(n_cols);

	int n = 0;
	if (n_rows == n_cols)
	{
		n = n_rows;
	}
	else
	{
		if (!extend_cost)
		{
			fprintf(stderr, "[lapjv] error: non-square cost matrix requires extend_cost=true\n");
			return -1.0;
		}
	}
		
	if (extend_cost || cost_limit < LONG_MAX)
	{
		n = n_rows + n_cols;
		cost_c_extended.resize(n);
		for (int i = 0; i < cost_c_extended.size(); i++)
			cost_c_extended[i].resize(n);

		if (cost_limit < LONG_MAX)
		{
			for (int i = 0; i < cost_c_extended.size(); i++)
			{
				for (int j = 0; j < cost_c_extended[i].size(); j++)
				{
					cost_c_extended[i][j] = cost_limit / 2.0;
				}
			}
		}
		else
		{
			float cost_max = -1;
			for (int i = 0; i < cost_c.size(); i++)
			{
				for (int j = 0; j < cost_c[i].size(); j++)
				{
					if (cost_c[i][j] > cost_max)
						cost_max = cost_c[i][j];
				}
			}
			for (int i = 0; i < cost_c_extended.size(); i++)
			{
				for (int j = 0; j < cost_c_extended[i].size(); j++)
				{
					cost_c_extended[i][j] = cost_max + 1;
				}
			}
		}

		for (int i = n_rows; i < cost_c_extended.size(); i++)
		{
			for (int j = n_cols; j < cost_c_extended[i].size(); j++)
			{
				cost_c_extended[i][j] = 0;
			}
		}
		for (int i = 0; i < n_rows; i++)
		{
			for (int j = 0; j < n_cols; j++)
			{
				cost_c_extended[i][j] = cost_c[i][j];
			}
		}

		cost_c.clear();
		cost_c.assign(cost_c_extended.begin(), cost_c_extended.end());
	}

	double **cost_ptr;
	cost_ptr = new double *[n];
	for (int i = 0; i < n; i++)
		cost_ptr[i] = new double[n];

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			cost_ptr[i][j] = cost_c[i][j];
		}
	}

	int* x_c = new int[sizeof(int) * n];
	int *y_c = new int[sizeof(int) * n];

	int ret = lapjv_internal(n, cost_ptr, x_c, y_c);
	if (ret != 0)
	{
		fprintf(stderr, "[lapjv] error: lapjv_internal returned %d\n", ret);
		for (int i = 0; i < n; i++) delete[] cost_ptr[i];
		delete[] cost_ptr;
		delete[] x_c;
		delete[] y_c;
		return -1.0;
	}

	double opt = 0.0;

	if (n != n_rows)
	{
		for (int i = 0; i < n; i++)
		{
			if (x_c[i] >= n_cols)
				x_c[i] = -1;
			if (y_c[i] >= n_rows)
				y_c[i] = -1;
		}
		for (int i = 0; i < n_rows; i++)
		{
			rowsol[i] = x_c[i];
		}
		for (int i = 0; i < n_cols; i++)
		{
			colsol[i] = y_c[i];
		}

		if (return_cost)
		{
			for (int i = 0; i < rowsol.size(); i++)
			{
				if (rowsol[i] != -1)
				{
					//cout << i << "\t" << rowsol[i] << "\t" << cost_ptr[i][rowsol[i]] << endl;
					opt += cost_ptr[i][rowsol[i]];
				}
			}
		}
	}
	else if (return_cost)
	{
		for (int i = 0; i < rowsol.size(); i++)
		{
			opt += cost_ptr[i][rowsol[i]];
		}
	}

	for (int i = 0; i < n; i++)
	{
		delete[]cost_ptr[i];
	}
	delete[]cost_ptr;
	delete[]x_c;
	delete[]y_c;

	return opt;
}

Scalar BYTETracker::get_color(int idx)
{

    Scalar color;
    switch (idx % 3)
    {
        case 0:
            color = Scalar(255, 0, 0); // 红色
            break;
        case 1:
            color = Scalar(0, 255, 0); // 绿色
            break;
        case 2:
            color = Scalar(0, 0, 255); // 蓝色
            break;
    }
    return color;
}
