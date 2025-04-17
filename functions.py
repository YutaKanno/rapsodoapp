import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import plotly.express as px
import numpy as np
from matplotlib.patches import Ellipse
import io
from datetime import datetime  




def make_movement_plot(rapsodo_file):

    # パレットの設定
    palette = {
        "Fastball": "#FF3333", "2-Seam": "#FF9933", "Slider": "#6666FF",
        "Cutter": "#9933FF", "CurveBall": "#66B2FF", "ChangeUp": "#00CC66",
        "Split": "#009900", "Sinker": "#CC00CC", "Shoot": "#FF66B2",
        "Other": "#000000"
    }


    # 平均球速の計算
    mean_speed = rapsodo_file.groupby("Pitch_Type")["球速"].mean().round(1).reset_index()
    mean_speed["label"] = mean_speed["Pitch_Type"] + " (" + mean_speed["球速"].astype(str) + " km/h)"

    # データにラベルをマージ
    rapsodo_with_label = rapsodo_file.merge(mean_speed, on="Pitch_Type", how="left")

    # カラーマッピング（ラベルと色を対応）
    palette_labeled = {row["label"]: palette[row["Pitch_Type"]] for _, row in mean_speed.iterrows()}

    # グラフ作成
    plt.figure(figsize=(8, 9))
    sns.scatterplot(
        data=rapsodo_with_label.dropna(subset=["HB_spin"]),  # 回転数が NaN でないデータを使用
        x="HB_spin", 
        y="VB_spin", 
        hue="label",  # 凡例を「球種 + 平均球速」に変更
        palette=palette_labeled,
        s=80  # 点のサイズ
    )
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.ylabel('Vertical Movment')
    plt.xlabel('Horizonal Movment')
    plt.legend(title="Movement Mapping", bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
    plt.subplots_adjust(bottom=0.3)
    # 画像をメモリに保存
    img = io.BytesIO()
    plt.savefig(img, format='png')
    #img.seek(0)  # ポインタを先頭に戻す

    # 画像を返す
    #return img

    plt.show()

#make_movement_plot(rapdata)


def make_rap_table(rap):
    rap['Spin_Direction'] = rap['Spin_Direction'].astype(str)

    # 文字列として処理し、strptimeを適用
    def safe_strptime(x):
        try:
            return datetime.strptime(x, '%H:%M')
        except ValueError:  # 時間の形式が間違っている場合はNoneを返す
            return None

    rap['time'] = rap['Spin_Direction'].apply(safe_strptime)
    rap['minute'] = rap['time'].apply(lambda x: x.hour * 60 + x.minute)
    rap['axis'] = np.where(
        (rap['Spin_Direction'] == '右') & (rap['minute'] > 680) & (rap['Pitch_Type'] == 'ストレート'),  
        720 - rap['minute'],  
        rap['minute']  
    )
    
    N = rap[rap['球速'] != 0].groupby('Pitch_Type').size()
    mean_velo = rap.groupby('Pitch_Type')['球速'].mean().round(1)
    max_velo = rap.groupby('Pitch_Type')['球速'].max().round(0)
    mean_spin = rap.groupby('Pitch_Type')['回転数'].mean().round(1)
    max_spin = rap.groupby('Pitch_Type')['回転数'].max().round(0)
    spin_eff = rap.groupby('Pitch_Type')['回転効率'].mean().round(1)
    mean_axis = rap.groupby('Pitch_Type')['axis'].mean().round(1)
    mean_axis_hm = mean_axis.apply(lambda x: f"{int(x // 60):02}:{int(x % 60):02}")
    mean_vb_spin = rap.groupby('Pitch_Type')['VB_spin'].mean().round(1)
    mean_hb_spin = rap.groupby('Pitch_Type')['HB_spin'].mean().round(1)
    mean_vb_traj = rap.groupby('Pitch_Type')['VB_trajectory'].mean().round(1)
    mean_hb_traj = rap.groupby('Pitch_Type')['HB_trajectory'].mean().round(1)
    
    rap_list = [N, mean_velo, max_velo, mean_spin, max_spin, spin_eff, mean_axis_hm, mean_vb_spin, mean_hb_spin, mean_vb_traj, mean_hb_traj]
    labels = ['N', 'Velo(Mean)', 'Velo(Max)', 'Spin(Mean)', 'Spin(Max)', 'Spin_Eff', 'Axis', 'VB(Spin)', 'HB(Spin)', 'VB(Traj)', 'HB(Traj)']
    df = pd.DataFrame({label: stat for label, stat in zip(labels, rap_list)})
    df = df.reset_index()
    df = df.sort_values(by='N', ascending=False)
    
    return df
    
    
    


def make_rap_table_by_date(rap):
    rap['Spin_Direction'] = rap['Spin_Direction'].astype(str)

    # 文字列として処理し、strptimeを適用
    def safe_strptime(x):
        try:
            return datetime.strptime(x, '%H:%M')
        except ValueError:  # 時間の形式が間違っている場合はNoneを返す
            return None

    rap['time'] = rap['Spin_Direction'].apply(safe_strptime)
    rap['minute'] = rap['time'].apply(lambda x: x.hour * 60 + x.minute)
    rap['axis'] = np.where(
        (rap['Spin_Direction'] == '右') & (rap['minute'] > 680) & (rap['Pitch_Type'] == 'ストレート'),  
        720 - rap['minute'],  
        rap['minute']  
    )
    
    N = rap[rap['球速'] != 0].groupby(['Pitch_Type', 'Date']).size()
    mean_velo = rap.groupby(['Pitch_Type', 'Date'])['球速'].mean().round(1)
    max_velo = rap.groupby(['Pitch_Type', 'Date'])['球速'].max().round(0)
    mean_spin = rap.groupby(['Pitch_Type', 'Date'])['回転数'].mean().round(1)
    max_spin = rap.groupby(['Pitch_Type', 'Date'])['回転数'].max().round(0)
    spin_eff = rap.groupby(['Pitch_Type', 'Date'])['回転効率'].mean().round(1)
    mean_axis = rap.groupby(['Pitch_Type', 'Date'])['axis'].mean().round(1)
    mean_axis_hm = mean_axis.apply(lambda x: f"{int(x // 60):02}:{int(x % 60):02}")
    mean_vb_spin = rap.groupby(['Pitch_Type', 'Date'])['VB_spin'].mean().round(1)
    mean_hb_spin = rap.groupby(['Pitch_Type', 'Date'])['HB_spin'].mean().round(1)
    mean_vb_traj = rap.groupby(['Pitch_Type', 'Date'])['VB_trajectory'].mean().round(1)
    mean_hb_traj = rap.groupby(['Pitch_Type', 'Date'])['HB_trajectory'].mean().round(1)
    
    rap_list = [N, mean_velo, max_velo, mean_spin, max_spin, spin_eff, mean_axis_hm, mean_vb_spin, mean_hb_spin, mean_vb_traj, mean_hb_traj]
    labels = ['N', 'Velo(Mean)', 'Velo(Max)', 'Spin(Mean)', 'Spin(Max)', 'Spin_Eff', 'Axis', 'VB(Spin)', 'HB(Spin)', 'VB(Traj)', 'HB(Traj)']
    df = pd.DataFrame({label: stat for label, stat in zip(labels, rap_list)})
    df = df.reset_index()
    df = df.sort_values(by=['Date', 'N'], ascending=[True, False])
    
    return df




def make_rap_plot_by_date(rapsodo_file, name):
    rapsodo_file = rapsodo_file[rapsodo_file['名前'] == name]
    # パレットの設定
    palette = ["#FF3333", "#FF9933", "#6666FF", "#9933FF", "#66B2FF",
            "#00CC66", "#009900", "#CC00CC", "#FF66B2", "#000000"]

    # 球種とパレットの対応関係
    pt = ["ストレート", "ツーシーム", "スライダー", "カット", "カーブ",
        "チェンジアップ", "フォーク", "シンカー", "シュート", "特殊球"]
    color_map = dict(zip(pt, palette))
    
    rapsodo_file = rapsodo_file[rapsodo_file['Pitch_Type'].isin(pt)]


    df_grouped = rapsodo_file.groupby(["球種", "日付"], as_index=False).agg(
        VB=("VB_spin", "mean"),
        HB=("HB_spin", "mean")
    )
    df_grouped['VB'] = df_grouped['VB'].round(1)
    df_grouped['HB'] = df_grouped['HB'].round(1)

    # Plotly で散布図を作成
    fig = px.scatter(
        df_grouped, x="HB", y="VB", color="球種", hover_data=["日付", 'VB', 'HB'],
        color_discrete_map=color_map,
        labels={"HB": "", "VB": ""},
        size_max=15,
        width=400,
        height=500
    )

    # 軸に基準線を追加
    fig.add_vline(x=0, line=dict(color="black", width=1))
    fig.add_hline(y=0, line=dict(color="black", width=1))

    # 凡例を非表示に設定
    fig.update_layout(
        showlegend=False
    )

    plot_html = fig.to_html(full_html=False)
    
    return plot_html




def make_density_plot(rap_file, kind):
    palette = {
        "Fastball": "#FF3333", "2-Seam": "#FF9933", "Slider": "#6666FF",
        "Cutter": "#9933FF", "CurveBall": "#66B2FF", "ChangeUp": "#00CC66",
        "Split": "#009900", "Sinker": "#CC00CC", "Shoot": "#FF66B2",
        "Other": "#000000"
    }


    # 平均球速の計算
    mean_speed = rap_file.groupby("Pitch_Type")[kind].mean().round(1).reset_index()
    mean_speed["label"] = mean_speed["Pitch_Type"] + " (" + mean_speed[kind].astype(str) + " )"

    # データにラベルをマージ
    rap_file = rap_file.merge(mean_speed[['Pitch_Type', 'label']], on="Pitch_Type", how="left")

    # カラーマッピング（ラベルと色を対応）
    palette_labeled = {row["label"]: palette[row["Pitch_Type"]] for _, row in mean_speed.iterrows()}

    # seabornで密度プロットを作成
    plt.figure(figsize=(15, 5))

    # 密度プロットを作成
    sns.kdeplot(data=rap_file, x=kind, hue='label', fill=True, alpha=0.3, palette=palette_labeled)

    # y軸を非表示に
    plt.gca().get_yaxis().set_visible(False)

    # 枠線を削除
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)



    # 画像を返す
    #return img

    return plt

#make_density_plot(rapdata)







def mov_plot(rapdata, kind):
    # パレットの設定
    palette = {
        "Fastball": "#FF3333", "2-Seam": "#FF9933", "Slider": "#6666FF",
        "Cutter": "#9933FF", "CurveBall": "#66B2FF", "ChangeUp": "#00CC66",
        "Split": "#009900", "Sinker": "#CC00CC", "Shoot": "#FF66B2",
        "Other": "#000000"
    }

    mean_hb = rapdata.groupby('Pitch_Type')[f'HB_{kind}'].mean().to_dict()
    mean_vb = rapdata.groupby('Pitch_Type')[f'VB_{kind}'].mean().to_dict()
    
    # mean_hbとmean_vbを結合した列を作成
    mean_combined = {pitch: f"{mean_hb[pitch]:.1f}-{mean_vb[pitch]:.1f}" for pitch in mean_hb.keys()}

    # プロット作成
    fig, ax = plt.subplots(figsize=(8,9))

    # 散布図
    sns.scatterplot(data=rapdata, x=f'HB_{kind}', y=f'VB_{kind}', hue='Pitch_Type', palette=palette, ax=ax)

    # 球種ごとに楕円を描画
    for Pitch_Type, group in rapdata.groupby('Pitch_Type'):
        mean = group[[f'HB_{kind}', f'VB_{kind}']].mean().values
        cov = group[[f'HB_{kind}', f'VB_{kind}']].cov().values
        
        if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
            pass
        else:
            eigvals, eigvecs = np.linalg.eig(cov)
            
            angle = np.arctan2(*eigvecs[:, 0][::-1]) * 180 / np.pi
            width, height = 2 * np.sqrt(eigvals)

            ellipse = Ellipse(mean, width=width, height=height, angle=angle, color=palette[Pitch_Type], alpha=0.3)
            ax.add_patch(ellipse)

    # 軸の設定
    ax.axhline(0, color='black', linestyle='--')
    ax.axvline(0, color='black', linestyle='--')
    ax.set_xlabel('Horizontal Movement')  # X軸ラベル
    ax.set_ylabel('Vertical Movement')  # Y軸ラベル

    # レジェンドに平均球速を追加
    handles, labels = ax.get_legend_handles_labels()
    new_labels = [f"{label} ({mean_combined.get(label, 'N/A')})" for label in labels]
    ax.legend(handles, new_labels, title='Pitch Type(HB(cm)-VB(cm))')

    return plt
    
    

def release_plot(rapdata):
    # パレットの設定
    palette = {
        "Fastball": "#FF3333", "2-Seam": "#FF9933", "Slider": "#6666FF",
        "Cutter": "#9933FF", "CurveBall": "#66B2FF", "ChangeUp": "#00CC66",
        "Split": "#009900", "Sinker": "#CC00CC", "Shoot": "#FF66B2",
        "Other": "#000000"
    }


    # プロット作成
    fig, ax = plt.subplots(figsize=(8,9))

    # 散布図
    sns.scatterplot(data=rapdata, x='Release_Side', y='Release_Height', hue='Pitch_Type', palette=palette, ax=ax)

    # 球種ごとに楕円を描画
    for Pitch_Type, group in rapdata.groupby('Pitch_Type'):
        mean = group[['Release_Side', 'Release_Height']].mean().values
        cov = group[['Release_Side', 'Release_Height']].cov().values
        
        if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
            pass
        else:
            eigvals, eigvecs = np.linalg.eig(cov)
            
            angle = np.arctan2(*eigvecs[:, 0][::-1]) * 180 / np.pi
            width, height = 2 * np.sqrt(eigvals)

            ellipse = Ellipse(mean, width=width, height=height, angle=angle, color=palette[Pitch_Type], alpha=0.3)
            ax.add_patch(ellipse)

    return plt

'''
def plot_mov_mean(name, this_season, rapsodo_file):
    # カラーパレットの設定
    palette = {
        "Fastball": "#FF3333", "2-Seam": "#FF9933", "Slider": "#6666FF",
        "Cutter": "#9933FF", "Curve": "#66B2FF", "Changeup": "#00CC66",
        "Split": "#009900", "Sinker": "#CC00CC", "Shoot": "#FF66B2",
        "Other": "#000000"
    }

    # 球種のマッピング
    Pitch_Type_mapping = {
        "ストレート": "Fastball", "ツーシーム": "2-Seam", "スライダー": "Slider",
        "カット": "Cutter", "カーブ": "Curve", "チェンジアップ": "Changeup",
        "フォーク": "Split", "シンカー": "Sinker", "シュート": "Shoot",
        "特殊球": "Other"
    }

    rapsodo_file["Pitch_Type"] = rapsodo_file["球種"].map(Pitch_Type_mapping)
    rapsodo_file["palette"] = rapsodo_file["Pitch_Type"].map(palette)

    # 全体の平均回転数と回転効率
    spin_mean = rapsodo_file.groupby(['Pitch_Type', 'palette'])['VB_spin'].mean()
    eff_mean = rapsodo_file.groupby(['Pitch_Type', 'palette'])['HB_spin'].mean()
    mean_df = pd.DataFrame({
        'spin_mean': spin_mean,
        'spin_eff_mean': eff_mean
    }).reset_index()

    rapdata = rapsodo_file[rapsodo_file['名前'] == name]
    spin_mean = rapdata.groupby(['Pitch_Type', 'palette'])['VB_spin'].mean()
    eff_mean = rapdata.groupby(['Pitch_Type', 'palette'])['HB_spin'].mean()
    mean_df_indiv = pd.DataFrame({
        'spin_mean': spin_mean,
        'spin_eff_mean': eff_mean
    }).reset_index()

    # データの結合
    combined_df = pd.merge(mean_df_indiv, mean_df, on=['Pitch_Type', 'palette'], suffixes=('_indiv', '_overall'))
    # プロット
    fig, ax = plt.subplots(figsize=(8, 9))

    # 散布図: 個人データ
    for i, Pitch_Type in enumerate(combined_df['Pitch_Type'].unique()):
        # 個別に散布図を描く
        Pitch_Type_data = combined_df[combined_df['Pitch_Type'] == Pitch_Type]
        ax.scatter(Pitch_Type_data['spin_mean_indiv'], Pitch_Type_data['spin_eff_mean_indiv'],
                color=Pitch_Type_data['palette'], label=Pitch_Type, alpha=0.7)
        # 座標をラベルとして表示
        for j, row in Pitch_Type_data.iterrows():
            ax.text(row['spin_mean_indiv'], row['spin_eff_mean_indiv']+3,
                    f"({row['spin_mean_indiv']:.1f}cm, {row['spin_eff_mean_indiv']:.1f}cm)",
                    fontsize=12, ha='center', va='center', color=row['palette'], alpha=0.8)
            
    # 散布図: 全体データ
    for i, Pitch_Type in enumerate(combined_df['Pitch_Type'].unique()):
        Pitch_Type_data = combined_df[combined_df['Pitch_Type'] == Pitch_Type]
        ax.scatter(Pitch_Type_data['spin_mean_overall'], Pitch_Type_data['spin_eff_mean_overall'],
                color=Pitch_Type_data['palette'], alpha=0.7)

    # 個人データの点に全体から矢印を引く
    for i in range(len(combined_df)):
        x1, y1 = combined_df['spin_mean_overall'].iloc[i], combined_df['spin_eff_mean_overall'].iloc[i]  # 全体データ
        x2, y2 = combined_df['spin_mean_indiv'].iloc[i], combined_df['spin_eff_mean_indiv'].iloc[i]  # 個人データ
        
        # 矢印を描く
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(facecolor=combined_df['palette'].iloc[i], edgecolor=combined_df['palette'].iloc[i], arrowstyle='->', lw=1))

    # 軸ラベル
    ax.set_xlabel('Horizonal Movment(cm)')
    ax.set_ylabel('Vertical Movement(cm)')
    ax.set_title('Movement')
    ax.legend(title='Pitch Type')
    ax.axhline(0, color='black', linestyle='--')
    ax.axvline(0, color='black', linestyle='--')


    # 画像をメモリに保存
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', pad_inches=0.1)
    img.seek(0)  # ポインタを先頭に戻す

    output_path = f'jinja2/folders/{name}_{this_season}/mov_means.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    '''
    
    



'''
def plot_spin(name, this_season, rapsodo_file):
    # カラーパレットの設定
    palette = {
        "Fastball": "#FF3333", "2-Seam": "#FF9933", "Slider": "#6666FF",
        "Cutter": "#9933FF", "Curve": "#66B2FF", "Changeup": "#00CC66",
        "Split": "#009900", "Sinker": "#CC00CC", "Shoot": "#FF66B2",
        "Other": "#000000"
    }

    # 球種のマッピング
    Pitch_Type_mapping = {
        "ストレート": "Fastball", "ツーシーム": "2-Seam", "スライダー": "Slider",
        "カット": "Cutter", "カーブ": "Curve", "チェンジアップ": "Changeup",
        "フォーク": "Split", "シンカー": "Sinker", "シュート": "Shoot",
        "特殊球": "Other"
    }

    rapsodo_file["Pitch_Type"] = rapsodo_file["球種"].map(Pitch_Type_mapping)
    rapsodo_file["palette"] = rapsodo_file["Pitch_Type"].map(palette)

    # 全体の平均回転数と回転効率
    spin_mean = rapsodo_file.groupby(['Pitch_Type', 'palette'])['回転数'].mean()
    eff_mean = rapsodo_file.groupby(['Pitch_Type', 'palette'])['回転効率'].mean()
    mean_df = pd.DataFrame({
        'spin_mean': spin_mean,
        'spin_eff_mean': eff_mean
    }).reset_index()

    rapdata = rapsodo_file[rapsodo_file['名前'] == name]
    spin_mean = rapdata.groupby(['Pitch_Type', 'palette'])['回転数'].mean()
    eff_mean = rapdata.groupby(['Pitch_Type', 'palette'])['回転効率'].mean()
    mean_df_indiv = pd.DataFrame({
        'spin_mean': spin_mean,
        'spin_eff_mean': eff_mean
    }).reset_index()

    # データの結合
    combined_df = pd.merge(mean_df_indiv, mean_df, on=['Pitch_Type', 'palette'], suffixes=('_indiv', '_overall'))
    # プロット
    fig, ax = plt.subplots(figsize=(15, 5))

    # 散布図: 個人データ
    for i, Pitch_Type in enumerate(combined_df['Pitch_Type'].unique()):
        # 個別に散布図を描く
        Pitch_Type_data = combined_df[combined_df['Pitch_Type'] == Pitch_Type]
        ax.scatter(Pitch_Type_data['spin_mean_indiv'], Pitch_Type_data['spin_eff_mean_indiv'],
                color=Pitch_Type_data['palette'], label=Pitch_Type, alpha=0.7)
        # 座標をラベルとして表示
        for j, row in Pitch_Type_data.iterrows():
            ax.text(row['spin_mean_indiv'], row['spin_eff_mean_indiv']+3,
                    f"({row['spin_mean_indiv']:.0f}rpm, {row['spin_eff_mean_indiv']:.1f}%)",
                    fontsize=12, ha='center', va='center', color=row['palette'], alpha=0.8)
            
    # 散布図: 全体データ
    for i, Pitch_Type in enumerate(combined_df['Pitch_Type'].unique()):
        Pitch_Type_data = combined_df[combined_df['Pitch_Type'] == Pitch_Type]
        ax.scatter(Pitch_Type_data['spin_mean_overall'], Pitch_Type_data['spin_eff_mean_overall'],
                color=Pitch_Type_data['palette'], alpha=0.7)

    # 個人データの点に全体から矢印を引く
    for i in range(len(combined_df)):
        x1, y1 = combined_df['spin_mean_overall'].iloc[i], combined_df['spin_eff_mean_overall'].iloc[i]  # 全体データ
        x2, y2 = combined_df['spin_mean_indiv'].iloc[i], combined_df['spin_eff_mean_indiv'].iloc[i]  # 個人データ
        
        # 矢印を描く
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(facecolor=combined_df['palette'].iloc[i], edgecolor=combined_df['palette'].iloc[i], arrowstyle='->', lw=1))

    # 軸ラベル
    ax.set_xlabel('Average Spin Rate')
    ax.set_ylabel('Average Spin Efficiency')
    ax.set_title('Spin Rate and Spin Eff')
    ax.legend(title='Pitch Type')

    # 画像をメモリに保存
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', pad_inches=0.1)
    img.seek(0)  # ポインタを先頭に戻す

    output_path = f'jinja2/folders/{name}_{this_season}/spin_plot.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

'''




'''
def plot_trans_velo(name, this_season):
    rapdata = pd.read_csv(f'jinja2/folders/{name}/rap_data.csv')
    # パレットの設定（球種ごとに色を定義）
    palette = {
        "ストレート": "#FF3333", "ツーシーム": "#FF9933", "スライダー": "#6666FF",
        "カット": "#9933FF", "カーブ": "#66B2FF", "チェンジアップ": "#00CC66",
        "フォーク": "#009900", "シンカー": "#CC00CC", "シュート": "#FF66B2", "特殊球": "#000000"
    }
    rapdata["init_date"] = pd.to_datetime(rapdata["init_date"])
    # データの集計（日付順にソート）
    plot_data = (
        rapdata.groupby(["球種", "init_date"], as_index=False)
        .agg(平均球速=("Velo(Mean)", "mean"))
        .sort_values(["球種", "init_date"])  # 日付順にソート
    )
    plot_data["yyyymm"] = plot_data["init_date"].dt.strftime("%Y/%m") 
    # プロットを作成
    fig = px.line(
        plot_data,
        x="yyyymm",
        y="平均球速",
        color="球種",
        color_discrete_map=palette,  # カラーを指定
        markers=True,  # マーカーを追加
        title="By Seasons",
        labels={"yyyymm": "Date", "平均球速": "Mean_Velo"},
    )

    # レイアウト調整
    fig.update_layout(
        xaxis=dict(title="Date"),
        yaxis=dict(title="Mean_Velo"),
    )
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="white",
    )

    plot_html = fig.to_html(full_html=False)
    
    return plot_html

'''
'''
def plot_trans_spin(name, this_season):
    rapdata = pd.read_csv(f'jinja2/folders/{name}/rap_data.csv')
    # パレットの設定（球種ごとに色を定義）
    palette = {
        "ストレート": "#FF3333", "ツーシーム": "#FF9933", "スライダー": "#6666FF",
        "カット": "#9933FF", "カーブ": "#66B2FF", "チェンジアップ": "#00CC66",
        "フォーク": "#009900", "シンカー": "#CC00CC", "シュート": "#FF66B2", "特殊球": "#000000"
    }
    rapdata["init_date"] = pd.to_datetime(rapdata["init_date"])
    # データの集計（日付順にソート）
    plot_data = (
        rapdata.groupby(["球種", "init_date"], as_index=False)
        .agg(平均回転効率=("Spin_Eff", "mean"),
             平均回転数 = ('Spin(Mean)', 'mean'))
        .sort_values(["球種", "init_date"])  # 日付順にソート
    )
    plot_data["yyyymm"] = plot_data["init_date"].dt.strftime("%Y/%m") 
    # プロットを作成
    fig = px.scatter(
        plot_data,
        y="平均回転効率",
        x="平均回転数",
        color="球種",
        color_discrete_map=palette,  # カラーを指定
        title="By Seasons",
        size_max=15,
        labels={"平均回転効率": "Mean_Spin_Eff", "平均回転数": "Mean_Spin_Rate"},
        hover_data={"yyyymm": True}
    )

    # レイアウト調整
    fig.update_layout(
        plot_bgcolor="white",
        yaxis=dict(title="Mean_Spin_Eff"),
        xaxis=dict(title="Mean_Spin_Rate"),
    )
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="white",
    )

    # グラフを表示
    plot_html = fig.to_html(full_html=False)
    
    return plot_html


'''

'''
def plot_trans_mov(name, this_season):
    
    rapdata = pd.read_csv(f'jinja2/folders/{name}/rap_data.csv')
    # パレットの設定（球種ごとに色を定義）
    palette = {
        "ストレート": "#FF3333", "ツーシーム": "#FF9933", "スライダー": "#6666FF",
        "カット": "#9933FF", "カーブ": "#66B2FF", "チェンジアップ": "#00CC66",
        "フォーク": "#009900", "シンカー": "#CC00CC", "シュート": "#FF66B2", "特殊球": "#000000"
    }
    rapdata["init_date"] = pd.to_datetime(rapdata["init_date"])
    # データの集計（日付順にソート）
    plot_data = (
        rapdata.groupby(["球種", "init_date"], as_index=False)
        .agg(縦=("VB", "mean"),
             横 = ('HB', 'mean'))
        .sort_values(["球種", "init_date"])  # 日付順にソート
    )
    plot_data["yyyymm"] = plot_data["init_date"].dt.strftime("%Y/%m") 
    # プロットを作成
    fig = px.scatter(
        plot_data,
        y="縦",
        x="横",
        color="球種",
        color_discrete_map=palette,  # カラーを指定
        title="By Seasons",
        labels={"縦": "VB", "横": "HB"},
        hover_data={"yyyymm": True},
        width=400,
        height=500
    )

    # レイアウト調整
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="white",
        yaxis=dict(title="VB"),
        xaxis=dict(title="HB"),
    )
    fig.add_vline(x=0, line=dict(color="black", width=1))
    fig.add_hline(y=0, line=dict(color="black", width=1))

    # グラフを表示
    plot_html = fig.to_html(full_html=False)
    
    return plot_html
'''



def plot_axis(df):
    # 特定の選手のデータを取得（明示的にコピー）
    df = df.dropna(subset=['Spin_Direction'])
    
    def time_to_degree(time_str):
        hh, mm = map(int, time_str.split(":"))
        minute = hh * 60 + mm  
        degree =  (minute /2) - 90 
        return degree % 360 

    df["角度"] = df["Spin_Direction"].astype(str).apply(time_to_degree)

    # 角度を15°ごとのグループに分ける
    bins = list(range(0, 361, 15))  # 0°から360°まで15°刻み
    labels = [f"{i}°-{i+15}°" for i in range(0, 360, 15)]  # ラベル作成
    df["角度グループ"] = pd.cut(df["角度"], bins=bins, labels=labels, right=False, include_lowest=True)

    # 球種ごとの出現頻度を集計
    df_grouped = df.groupby(["Pitch_Type", "角度グループ"]).size().reset_index(name="frequency")

    # パレット（球種ごとの色）
    palette = {
        "Fastball": "#FF3333", "2-Seam": "#FF9933", "Slider": "#6666FF",
        "Cutter": "#9933FF", "CurveBall": "#66B2FF", "ChangeUp": "#00CC66",
        "Split": "#009900", "Sinker": "#CC00CC", "Shoot": "#FF66B2",
        "Other": "#000000"
    }

    # プロット作成
    fig = px.bar_polar(
        df_grouped,
        r="frequency",  # 出現頻度
        theta="角度グループ",  # 15°ごとのグループ
        color="Pitch_Type",
        color_discrete_map=palette, 
        title="Spin Axis",
        start_angle=0,  # 12:00を0度に設定
        direction="clockwise",  # 時計回りにする（デフォルトは反時計回り）
        width=400,  # グラフの幅を調整
        height=500,  # グラフの高さを調整
        barmode="group",  # 棒の幅を広げる
        category_orders={"Pitch_Type": list(palette.keys())}  # 球種順の指定
    )
    fig.update_layout(
        plot_bgcolor="white",
        legend=dict(
            orientation="h",  # 凡例を横に配置
            x=0.5,  # 凡例の位置を中央に
            xanchor="center",  # 中央に配置
            y=-0.2,  # グラフの下に配置
            yanchor="bottom"  # 下に配置
        )
    )
    
    fig.update_layout(
        polar=dict(
            angularaxis=dict(
                showticklabels=False,  # 軸の数字を非表示に
            )
        )
    )

    return fig

 
 
 
def zone_plot(rapdata, c):
    if c == 'c':
        rapdata['Side'] = rapdata['Strike_Zone_Side']*(-1)
    else:
        rapdata['Side'] = rapdata['Strike_Zone_Side']
        
    
    rapdata = rapdata[(rapdata['Side'] >= -50) & (rapdata['Side'] <= 50)]
    rapdata = rapdata[(rapdata['Strike_Zone_Height'] >= 0) & (rapdata['Strike_Zone_Height'] <= 160)]
    # 散布図を作成
    fig, ax = plt.subplots(figsize=(8, 9))
    sns.scatterplot(
        data=rapdata,
        x="Side",
        y="Strike_Zone_Height",
        hue="Is_Strike",
        palette="coolwarm",
        ax=ax,
        s=65 # ポイントサイズを大きく設定
    )
    # 軌跡を描画
    trajectory_x = [25, 25, -25, -25, 25]
    trajectory_y = [46, 107, 107, 46, 46]
    ax.plot(trajectory_x, trajectory_y, color="green", linestyle="-", linewidth=1.5)
    ax.legend()
    # ストライクゾーンの枠を描画
    strike_zone = plt.Rectangle((-0.83, 1.5), 1.66, 2.0, fill=False, color="black", linestyle="--", linewidth=1.5)
    ax.add_patch(strike_zone)
    # 軸ラベルと数字を非表示に
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])

    return plt
  
  
  

def trend_plot(rapdata, kind):
    palette = {
        "Fastball": "#FF3333", "2-Seam": "#FF9933", "Slider": "#6666FF",
        "Cutter": "#9933FF", "CurveBall": "#66B2FF", "ChangeUp": "#00CC66",
        "Split": "#009900", "Sinker": "#CC00CC", "Shoot": "#FF66B2",
        "Other": "#000000"
    }
    
    # 日付ごとにkind列の平均値を計算
    trend_data = rapdata.groupby(['Date', 'Pitch_Type'])[kind].mean().reset_index()

    # プロット作成
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=trend_data,
        x='Date',
        y=kind,
        hue='Pitch_Type',
        palette=palette,
        marker='o'
    )

    # 軸ラベルとタイトルの設定
    plt.xlabel('Date')
    plt.ylabel(f'Average')
    plt.legend(title='Pitch Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    
    # 枠線を削除
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    
    return plt