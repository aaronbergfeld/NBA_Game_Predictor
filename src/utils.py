import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import pandas as pd
import time
import pickle

def get_injury_report():
    chrome_options = Options()
    # chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    driver = webdriver.Chrome(options=chrome_options)
    driver.get('https://espn.com/nba/injuries')
    # Click out of popup
    try:
        driver.find_element(By.CLASS_NAME, 'ab-close-button').click()
    except:
        pass
    injury_report = pd.DataFrame(columns=['PLAYER_ID', 'NAME', 'TEAM', 'STATUS', 'POS', 'RETURN_DATE'])
    # Find all tables with class 'ResponsiveTable Table__league-injuries'
    tables = driver.find_elements(By.CLASS_NAME, 'ResponsiveTable.Table__league-injuries')
    for table in tables:
        team_name = table.find_element(By.CLASS_NAME, 'injuries__teamName').text
        # Find all rows in the table
        rows = table.find_elements(By.CLASS_NAME, 'Table__TR.Table__TR--sm.Table__even')
        for row in rows:
            player_name = row.find_element(By.CLASS_NAME, 'col-name.Table__TD').text
            try:
                id = PLAYER_TO_PLAYER_ID[player_name]
            except KeyError:
                print(f'Could not find player {player_name}')
                continue
            status = row.find_element(By.CLASS_NAME, 'col-stat').find_element(By.TAG_NAME, 'span').text
            return_date = row.find_element(By.CLASS_NAME, 'col-date').text
            position = row.find_element(By.CLASS_NAME, 'col-pos').text
            injury_report = injury_report._append({
                'PLAYER_ID': id,
                'NAME': player_name,
                'TEAM': team_name,
                'STATUS': status,
                'POS': position,
                'RETURN_DATE': return_date
            }, ignore_index=True)
    driver.quit()
    return injury_report

def standardize_data(model_version, player_data, team_data):
    with open(f'../models/{model_version}_params.pkl', 'rb') as f:
        optimal_params = pickle.load(f)
        player_mean = pickle.load(f)
        player_std = pickle.load(f)
        team_mean = pickle.load(f)
        team_std = pickle.load(f)
    
    player_data_standardized = player_data.copy()
    player_data_standardized = (player_data_standardized - player_mean) / player_std
    team_data_standardized = team_data.copy()
    team_data_standardized = (team_data_standardized - team_mean) / team_std
    return player_data_standardized, team_data_standardized

def get_optimal_params(model_version):
    with open(f'../models/{model_version}_params.pkl', 'rb') as f:
        optimal_params = pickle.load(f)
    return optimal_params

if __name__ == '__main__':
    print(get_injury_report().head())
def load_data():
    games = pd.read_csv('../data/games.csv')
    players_deep = pd.read_csv('../data/rolling_averages_deep.csv')
    players_wide = pd.read_csv('../data/rolling_averages_wide.csv')
    odds = pd.read_csv('../data/odds.csv')
    team_level_data = pd.read_csv('../data/team_level_data.csv')
    injury_report = get_injury_report()
    
    injury_report = injury_report[injury_report['STATUS'] == 'Out']

    players_deep = players_deep.groupby('GAME_ID').filter(lambda x: len(x['TEAM_ID'].unique()) == 2)

    players_wide = players_wide.groupby('GAME_ID').filter(lambda x: len(x['TEAM_ID'].unique()) == 2)

    games = games[games['GAME_ID'].isin(players_deep['GAME_ID'].unique())]
    return games, players_deep, players_wide, team_level_data, odds, injury_report

TEAM_NAME_TO_ID = {
    'Boston Celtics': 1610612738, 
    'Portland Trail Blazers': 1610612757, 
    'Phoenix Suns': 1610612756, 
    'Miami Heat': 1610612748, 
    'Houston Rockets': 1610612745, 
    'Los Angeles Lakers': 1610612747, 
    'New Orleans Pelicans': 1610612740, 
    'Milwaukee Bucks': 1610612749, 
    'Cleveland Cavaliers': 1610612739, 
    'Detroit Pistons': 1610612765, 
    'Brooklyn Nets': 1610612751, 
    'Toronto Raptors': 1610612761, 
    'New York Knicks': 1610612752, 
    'Los Angeles Clippers': 1610612746, 
    'San Antonio Spurs': 1610612759, 
    'Indiana Pacers': 1610612754, 
    'Sacramento Kings': 1610612758, 
    'Minnesota Timberwolves': 1610612750, 
    'Chicago Bulls': 1610612741, 
    'Charlotte Hornets': 1610612766, 
    'Dallas Mavericks': 1610612742, 
    'Denver Nuggets': 1610612743, 
    'Memphis Grizzlies': 1610612763, 
    'Golden State Warriors': 1610612744, 
    'Atlanta Hawks': 1610612737, 
    'Philadelphia 76ers': 1610612755, 
    'Oklahoma City Thunder': 1610612760, 
    'Utah Jazz': 1610612762, 
    'Orlando Magic': 1610612753, 
    'Washington Wizards': 1610612764
}

PLAYER_TO_PLAYER_ID = {'Romeo Langford': 1629641, 'Jeremy Sochan': 1631110, 'Jakob Poeltl': 1627751, 'Devin Vassell': 1630170, 'Tre Jones': 1630200, 'Zach Collins': 1628380, 'Doug McDermott': 203926, 'Josh Richardson': 1626196, 'Malaki Branham': 1631103, 'Keita Bates-Diop': 1628966, 'Charles Bassey': 1629646, 'Stanley Johnson': 1626169, 'Isaiah Roby': 1629676, 'Gorgui Dieng': 203476, 'Trey Murphy III': 1630530, 'Naji Marshall': 1630230, 'Jonas Valanciunas': 202685, 'Herbert Jones': 1630529, 'CJ McCollum': 203468, 'Willy Hernangomez': 1626195, 'Dyson Daniels': 1630700, 'Jose Alvarado': 1630631, 'Jaxson Hayes': 1629637, "Devonte' Graham": 1628984, 'Garrett Temple': 202066, 'Corey Kispert': 1630557, 'Kyle Kuzma': 1628398, 'Daniel Gafford': 1629655, 'Bradley Beal': 203078, 'Monte Morris': 1628420, 'Taj Gibson': 201959, 'Rui Hachimura': 1629060, 'Jordan Goodwin': 1630692, 'Will Barton': 203115, 'Devon Dotson': 1629653, 'Johnny Davis': 1631098, 'Anthony Gill': 1630264, 'Lauri Markkanen': 1628374, 'Jarred Vanderbilt': 1629020, 'Walker Kessler': 1631117, 'Jordan Clarkson': 203903, 'Mike Conley': 201144, 'Collin Sexton': 1629012, 'Rudy Gay': 200752, 'Malik Beasley': 1627736, 'Nickeil Alexander-Walker': 1629638, 'Leandro Bolmaro': 1630195, 'Ochai Agbaji': 1630534, 'Simone Fontecchio': 1631323, 'Udoka Azubuike': 1628962, 'Talen Horton-Tucker': 1629659, 'Micah Potter': 1630695, 'MarJon Beauchamp': 1630699, 'Giannis Antetokounmpo': 203507, 'Brook Lopez': 201572, 'Grayson Allen': 1628960, 'Jrue Holiday': 201950, 'Bobby Portis': 1626171, 'Pat Connaughton': 1626192, 'Joe Ingles': 204060, 'Jevon Carter': 1628975, 'Wesley Matthews': 202083, 'Thanasis Antetokounmpo': 203648, 'George Hill': 201588, 'Serge Ibaka': 201586, 'Jordan Nwora': 1629670, 'Isaac Okoro': 1630171, 'Evan Mobley': 1630596, 'Jarrett Allen': 1628386, 'Donovan Mitchell': 1628378, 'Darius Garland': 1629636, 'Kevin Love': 201567, 'Caris LeVert': 1627747, 'Cedi Osman': 1626224, 'Mamadi Diakite': 1629603, 'Robin Lopez': 201577, 'Raul Neto': 203526, 'Bojan Bogdanovic': 202711, 'Isaiah Stewart': 1630191, 'Jalen Duren': 1631105, 'Jaden Ivey': 1631093, 'Killian Hayes': 1630165, 'Marvin Bagley III': 1628963, 'Alec Burks': 202692, 'Saddiq Bey': 1630180, 'Kevin Knox II': 1628995, 'Hamidou Diallo': 1628977, 'Rodney McGruder': 203585, 'Cory Joseph': 202709, 'Nerlens Noel': 203457, 'Tobias Harris': 202699, 'P.J. Tucker': 200782, 'Joel Embiid': 203954, "De'Anthony Melton": 1629001, 'James Harden': 201935, 'Georges Niang': 1627777, 'Montrezl Harrell': 1626149, 'Matisse Thybulle': 1629680, 'Shake Milton': 1629003, 'Danuel House Jr.': 1627863, 'Paul Reed': 1630194, 'Furkan Korkmaz': 1627788, 'DeMar DeRozan': 201942, 'Patrick Williams': 1630172, 'Nikola Vucevic': 202696, 'Zach LaVine': 203897, 'Alex Caruso': 1627936, 'Coby White': 1629632, 'Ayo Dosunmu': 1630245, 'Andre Drummond': 203083, 'Dalen Terry': 1631207, 'Tony Bradley': 1628396, 'Malcolm Hill': 1630792, "De'Andre Hunter": 1629631, 'John Collins': 1628381, 'Onyeka Okongwu': 1630168, 'Dejounte Murray': 1627749, 'Trae Young': 1629027, 'Bogdan Bogdanovic': 203992, 'AJ Griffin': 1631100, 'Aaron Holiday': 1628988, 'Frank Kaminsky': 1626163, 'Jalen Johnson': 1630552, 'Trent Forrest': 1630235, 'Justin Holiday': 203200, 'Vit Krejci': 1630249, 'Tyrese Martin': 1631213, 'Buddy Hield': 1627741, 'Aaron Nesmith': 1630174, 'Myles Turner': 1626167, 'Andrew Nembhard': 1629614, 'Tyrese Haliburton': 1630169, 'Bennedict Mathurin': 1631097, 'Chris Duarte': 1630537, 'Jalen Smith': 1630188, 'Oshae Brissett': 1629052, 'T.J. McConnell': 204456, 'Goga Bitadze': 1629048, 'Isaiah Jackson': 1630543, 'James Johnson': 201949, 'Jaylen Brown': 1627759, 'Jayson Tatum': 1628369, 'Al Horford': 201143, 'Derrick White': 1628401, 'Payton Pritchard': 1630202, 'Sam Hauser': 1630573, 'Grant Williams': 1629684, 'Malcolm Brogdon': 1627763, 'Robert Williams III': 1629057, 'Luke Kornet': 1628436, 'Blake Griffin': 201933, 'Justin Jackson': 1628382, 'Noah Vonleh': 203943, 'Jonathan Kuminga': 1630228, 'Draymond Green': 203110, 'Kevon Looney': 1626172, 'Moses Moody': 1630541, 'Jordan Poole': 1629673, 'Anthony Lamb': 1630237, 'Ty Jerome': 1629660, 'James Wiseman': 1630164, 'Patrick Baldwin Jr.': 1631116, 'Ryan Rollins': 1631157, 'Klay Thompson': 202691, 'Kevin Durant': 201142, "Royce O'Neale": 1626220, 'Nic Claxton': 1629651, 'Joe Harris': 203925, 'Ben Simmons': 1627732, 'Edmond Sumner': 1628410, 'Seth Curry': 203552, 'T.J. Warren': 203933, 'Yuta Watanabe': 1629139, 'Cam Thomas': 1630560, 'Patty Mills': 201988, 'Markieff Morris': 202693, 'Kyrie Irving': 202681, 'O.G. Anunoby': 1628384, 'Pascal Siakam': 1627783, 'Juancho Hernangomez': 1627823, 'Scottie Barnes': 1630567, 'Fred VanVleet': 1627832, 'Malachi Flynn': 1630201, 'Chris Boucher': 1628449, 'Thaddeus Young': 201152, 'Christian Koloko': 1631132, 'Dalano Banton': 1630625, 'Khem Birch': 203920, 'Gary Trent Jr.': 1629018, 'RJ Barrett': 1629628, 'Julius Randle': 203944, 'Mitchell Robinson': 1629011, 'Jalen Brunson': 1628973, 'Immanuel Quickley': 1630193, 'Miles McBride': 1630540, 'Isaiah Hartenstein': 1628392, 'Jericho Sims': 1630579, 'Derrick Rose': 201565, 'Ryan Arcidiacono': 1627853, 'Evan Fournier': 203095, 'Quentin Grimes': 1629656, 'Svi Mykhailiuk': 1629004, 'Cam Reddish': 1629629, 'Paolo Banchero': 1631094, 'Bol Bol': 1629626, 'Moritz Wagner': 1629021, 'Franz Wagner': 1630532, 'Markelle Fultz': 1628365, 'Terrence Ross': 203082, 'Mo Bamba': 1628964, 'Cole Anthony': 1630175, 'Admiral Schofield': 1629678, 'Kevon Harris': 1630284, 'R.J. Hampton': 1630181, 'Caleb Houstan': 1631216, 'Eric Gordon': 201569, 'Jabari Smith Jr.': 1631095, 'Alperen Sengun': 1630578, 'Jalen Green': 1630224, 'Kevin Porter Jr.': 1629645, 'Kenyon Martin Jr.': 1630231, 'Usman Garuba': 1630586, 'Tari Eason': 1631106, 'Daishen Nix': 1630227, 'Bruno Fernando': 1628981, 'Boban Marjanovic': 1626246, 'TyTy Washington Jr.': 1631102, 'Tim Hardaway Jr.': 203501, 'Reggie Bullock': 203493, 'Christian Wood': 1626174, 'Spencer Dinwiddie': 203915, 'Luka Doncic': 1629029, 'Dwight Powell': 203939, 'Kemba Walker': 202689, 'Frank Ntilikina': 1628373, 'Davis Bertans': 202722, 'JaVale McGee': 201580, 'Theo Pinson': 1629033, 'Anthony Edwards': 1630162, 'Jaden McDaniels': 1630183, 'Rudy Gobert': 203497, 'Austin Rivers': 203085, "D'Angelo Russell": 1626156, 'Naz Reid': 1629675, 'Jaylen Nowell': 1629669, 'Bryn Forbes': 1627854, 'Wendell Moore Jr.': 1631111, 'Luka Garza': 1630568, 'Nathan Knight': 1630233, 'Josh Minott': 1631169, 'Matt Ryan': 1630346, 'Josh Hart': 1628404, 'Jerami Grant': 203924, 'Jusuf Nurkic': 203994, 'Anfernee Simons': 1629014, 'Damian Lillard': 203081, 'Drew Eubanks': 1629234, 'Justise Winslow': 1626159, 'Shaedon Sharpe': 1631101, 'Trendon Watford': 1630570, 'Keon Johnson': 1630553, 'Greg Brown III': 1630535, 'Jabari Walker': 1631133, 'Luguentz Dort': 1629652, 'Jalen Williams': 1631114, 'Aleksej Pokusevski': 1630197, 'Josh Giddey': 1630581, 'Shai Gilgeous-Alexander': 1628983, 'Isaiah Joe': 1630198, 'Mike Muscala': 203488, 'Lindy Waters III': 1630322, 'Kenrich Williams': 1629026, 'Aaron Wiggins': 1630598, 'Eugene Omoruyi': 1630647, 'Darius Bazley': 1629647, 'Lonnie Walker IV': 1629022, 'LeBron James': 2544, 'Thomas Bryant': 1628418, 'Patrick Beverley': 201976, 'Dennis Schroder': 203471, 'Max Christie': 1631108, 'Troy Brown Jr.': 1628972, 'Wenyen Gabriel': 1629117, 'Damian Jones': 1627745, 'Kendrick Nunn': 1629134, 'Austin Reaves': 1630559, 'Juan Toscano-Anderson': 1629308, 'Harrison Barnes': 203084, 'Keegan Murray': 1631099, 'Domantas Sabonis': 1627734, 'Kevin Huerter': 1628989, "De'Aaron Fox": 1628368, 'Davion Mitchell': 1630558, 'Malik Monk': 1628370, 'Terence Davis': 1629056, 'Trey Lyles': 1626168, 'Neemias Queta': 1629674, 'Matthew Dellavedova': 203521, 'KZ Okpala': 1629644, 'Richaun Holmes': 1626158, 'Alex Len': 203458, 'Chimezie Metu': 1629002, 'Gordon Hayward': 202330, 'P.J. Washington': 1629023, 'Mason Plumlee': 203486, 'Kelly Oubre Jr.': 1626162, 'LaMelo Ball': 1630163, 'Jalen McDaniels': 1629667, 'Theo Maledon': 1630177, 'Nick Richards': 1630208, 'JT Thor': 1630550, 'Kai Jones': 1630539, 'James Bouknight': 1630547, 'Kawhi Leonard': 202695, 'Marcus Morris Sr.': 202694, 'Ivica Zubac': 1627826, 'Paul George': 202331, 'Reggie Jackson': 202704, 'John Wall': 202322, 'Nicolas Batum': 201587, 'Luke Kennard': 1628379, 'Norman Powell': 1626181, 'Terance Mann': 1629611, 'Amir Coffey': 1629599, 'Robert Covington': 203496, 'Moses Brown': 1629650, 'Derrick Jones Jr.': 1627884, 'Goran Dragic': 201609, 'Max Strus': 1629622, 'Haywood Highsmith': 1629312, 'Bam Adebayo': 1628389, 'Tyler Herro': 1629639, 'Victor Oladipo': 203506, 'Duncan Robinson': 1629130, 'Jamal Cain': 1631288, 'Dewayne Dedmon': 203473, 'Nikola Jovic': 1631107, 'Orlando Robinson': 1631115, 'Udonis Haslem': 2617, 'Deni Avdija': 1630166, 'Kristaps Porzingis': 204001, 'Mikal Bridges': 1628969, 'Torrey Craig': 1628470, 'Deandre Ayton': 1629028, 'Damion Lee': 1627814, 'Chris Paul': 101108, 'Ish Wainright': 1630688, 'Duane Washington Jr.': 1630613, 'Landry Shamet': 1629013, 'Bismack Biyombo': 202687, 'Devin Booker': 1626164, 'Dario Saric': 203967, 'Dillon Brooks': 1628415, 'Jaren Jackson Jr.': 1628991, 'Steven Adams': 203500, 'John Konchar': 1629723, 'Ja Morant': 1629630, 'Santi Aldama': 1630583, 'Tyus Jones': 1626145, 'David Roddy': 1631223, 'Brandon Clarke': 1629634, 'Ziaire Williams': 1630533, 'Xavier Tillman': 1630214, 'Jake LaRavia': 1631222, 'Kennedy Chandler': 1631113, 'Christian Braun': 1631128, 'Aaron Gordon': 203932, 'Nikola Jokic': 203999, 'Kentavious Caldwell-Pope': 203484, 'Bruce Brown': 1628971, 'Vlatko Cancar': 1628427, 'Jeff Green': 201145, 'Bones Hyland': 1630538, 'Zeke Nnaji': 1630192, 'Ish Smith': 202397, 'Davon Reed': 1628432, 'DeAndre Jordan': 201599, 'Lamar Stevens': 1630205, 'Jaden Springer': 1630531, 'Dorian Finney-Smith': 1627827, 'Kyle Anderson': 203937, 'Zion Williamson': 1629627, 'Gary Payton II': 1627780, 'Scotty Pippen Jr.': 1630590, 'Russell Westbrook': 201566, 'Josh Okogie': 1629006, 'Marcus Smart': 203935, 'Terry Taylor': 1630678, 'Donte DiVincenzo': 1628978, 'JaMychal Green': 203210, 'Precious Achiuwa': 1630173, 'Terry Rozier': 1626179, 'Bryce McGowens': 1631121, 'Jamal Murray': 1627750, 'Vernon Carey Jr.': 1630176, 'Moussa Diabate': 1631217, 'Brandon Boston Jr.': 1630527, 'Jason Preston': 1630554, 'Jimmy Butler': 202710, 'Keldon Johnson': 1629640, 'Jaden Hardy': 1630702, 'Kelly Olynyk': 203482, 'Sandro Mamukelashvili': 1630572, 'AJ Green': 1631260, 'Tre Mann': 1630544, 'Jaylin Williams': 1631119, 'Larry Nance Jr.': 1626204, 'Kira Lewis Jr.': 1630184, 'Jock Landale': 1629111, 'Jarrett Culver': 1629633, 'Keon Ellis': 1631165, 'JD Davison': 1631120, 'Jeff Dowtin Jr.': 1630288, 'Javonte Green': 1629750, 'Anthony Davis': 203076, 'Caleb Martin': 1628997, 'Garrison Mathews': 1629726, 'Josh Christopher': 1630528, 'Trevor Hudgins': 1631309, 'Khris Middleton': 203114, 'Stephen Curry': 201939, 'Clint Capela': 203991, 'Kyle Lowry': 200768, 'Dominick Barlow': 1631230, 'Isaiah Mobley': 1630600, 'Tyler Dorsey': 1628416, 'McKinley Wright IV': 1630593, 'Cameron Payne': 1626166, 'Mfiondu Kabengele': 1629662, "Day'Ron Sharpe": 1630549, 'Kenneth Lofton Jr.': 1631254, 'Jeremiah Robinson-Earl': 1630526, 'Maxi Kleber': 1628467, 'Alize Johnson': 1628993, 'Dru Smith': 1630696, 'Kessler Edwards': 1630556, 'David Duke Jr.': 1630561, 'Alondes Williams': 1631214, 'Isaiah Todd': 1630225, 'Peyton Watson': 1631212, 'Jordan McLaughlin': 1629162, 'Josh Green': 1630182, 'Obi Toppin': 1630167, 'Justin Champagnie': 1630551, 'Ron Harper Jr.': 1631199, 'Kendall Brown': 1631112, 'Trevelin Queen': 1630243, 'Dennis Smith Jr.': 1628372, 'Ousmane Dieng': 1631172, 'Saben Lee': 1630240, 'Gabe Vincent': 1629216, 'Vince Williams Jr.': 1631246, 'Jack White': 1631298, 'A.J. Lawson': 1630639, 'Andrew Wiggins': 203952, 'Ibou Badji': 1630641, 'Dean Wade': 1629731, 'Isaiah Livers': 1630587, 'Gary Harris': 203914, 'Dereon Seabron': 1631220, 'John Butler Jr.': 1631219, 'Nassir Little': 1629642, 'Julian Champagnie': 1630577, 'Karl-Anthony Towns': 1626157, 'Chuma Okeke': 1629643, 'Braxton Key': 1630296, 'Facundo Campazzo': 1630267, 'Jordan Hall': 1631160, 'Mark Williams': 1631109, 'Jalen Suggs': 1630591, 'Brandon Ingram': 1627742, 'Taurean Prince': 1627752, 'Kostas Antetokounmpo': 1628961, 'Marko Simonovic': 1630250, 'Michael Foster Jr.': 1630701, 'Chima Moneke': 1631320, 'Michael Porter Jr.': 1629008, 'Otto Porter Jr.': 203490, 'Tyrese Maxey': 1630178, 'Wendell Carter Jr.': 1628976, 'Jordan Schakel': 1630648, 'Desmond Bane': 1630217, 'Cade Cunningham': 1630595, 'Cameron Johnson': 1629661, 'Dylan Windler': 1629685, 'Blake Wesley': 1631104, "Jae'Sean Tate": 1630256, 'Johnny Juzang': 1630548, 'Delon Wright': 1626153, 'Joshua Primo': 1630563, 'Buddy Boeheim': 1631205, 'Trevor Keels': 1631211, 'Feron Hunt': 1630624, 'Cody Martin': 1628998, 'Cole Swider': 1631306, 'Reggie Bullock Jr.': 203493, 'KJ Martin': 1630231, 'Darius Days': 1630620, 'OG Anunoby': 1628384, 'Carlik Jones': 1630637, 'Sterling Brown': 1628425, 'Andre Iguodala': 2738, 'Joe Wieskamp': 1630580, 'PJ Dozier': 1628408, 'Jared Rhoden': 1631197, 'Derrick Favors': 202324, 'Ricky Rubio': 201937, 'E.J. Liddell': 1630604, 'Jonathan Isaac': 1628371, 'Donovan Williams': 1631495, 'Deonte Burton': 1629126, 'Danny Green': 201980, 'Daniel Theis': 1628464, 'Chris Silva': 1629735, 'Jamaree Bouyea': 1631123, 'DaQuan Jeffries': 1629610, 'Stanley Umude': 1630649, 'Quenton Jackson': 1631245, 'Kris Dunn': 1627739, 'Frank Jackson': 1628402, 'Olivier Sarr': 1630846, 'Cody Zeller': 203469, 'Jae Crowder': 203109, 'Meyers Leonard': 203086, 'Michael Carter-Williams': 203487, 'Willie Cauley-Stein': 1626161, 'Jared Butler': 1630215, 'Lester Quinones': 1631311, 'Sam Merrill': 1630241, 'Omer Yurtseven': 1630209, 'Lindell Wigginton': 1629623, 'Jay Huff': 1630643, 'Jarrell Brantley': 1629714, 'Xavier Cooks': 1641645, 'Xavier Moon': 1629875, 'D.J. Augustin': 201571, 'Jay Scrubb': 1630206, 'Xavier Sneed': 1630270, 'Luka Samanic': 1629677, 'Kobi Simmons': 1628424, 'Skylar Mays': 1630219, 'Shaquille Harrison': 1627885, 'Nate Williams': 1631466, 'Justin Minaya': 1631303, 'Gabe York': 1628221, 'Mac McClung': 1630644, 'Louis King': 1629663, 'Chance Comanche': 1628435, 'Jacob Gilyard': 1631367, 'RaiQuan Gray': 1630564, 'Trayce Jackson-Davis': 1631218, 'Brandin Podziemski': 1641764, 'Maxwell Lewis': 1641721, 'Collin Gillespie': 1631221, 'Jalen Pickett': 1629618, 'Julian Strawther': 1631124, 'Amen Thompson': 1641708, 'Cam Whitmore': 1641715, 'Jermaine Samuels Jr.': 1631257, 'Anthony Black': 1641710, 'Jett Howard': 1641724, 'Chet Holmgren': 1631096, 'Cason Wallace': 1641717, 'Vasilije Micic': 203995, 'Julian Phillips': 1641763, 'Leonard Miller': 1631159, 'Gradey Dick': 1641711, 'Emoni Bates': 1641734, 'Tristan Thompson': 202684, 'Noah Clowney': 1641730, 'Harry Giles III': 1628385, 'Ausar Thompson': 1641709, 'Marcus Sasser': 1631204, 'Malcolm Cazalon': 1630608, 'Jaime Jaquez Jr.': 1631170, 'Jordan Hawkins': 1641722, 'Kaiser Gates': 1629232, 'GG Jackson': 1641713, 'Jordan Walsh': 1641775, 'Sasha Vezenkov': 1628426, 'Jordan Ford': 1630259, 'Colby Jones': 1641732, 'Keyonte George': 1641718, 'Taylor Hendricks': 1641707, 'Brice Sensabaugh': 1641729, 'Kobe Bufkin': 1641723, 'Mouhamed Gueye': 1631243, 'Brandon Miller': 1641706, 'Nick Smith Jr.': 1641733, 'Amari Bailey': 1641735, 'Leaky Black': 1641778, 'Danilo Gallinari': 201568, 'Bilal Coulibaly': 1641731, 'Jarace Walker': 1641716, 'Ben Sheppard': 1641767, 'Dereck Lively II': 1641726, 'Dante Exum': 203957, 'Olivier-Maxence Prosper': 1641765, 'Victor Wembanyama': 1641705, 'Scoot Henderson': 1630703, 'Toumani Camara': 1641739, 'Kris Murray': 1631200, 'Rayan Rupert': 1641712, 'Kobe Brown': 1641738, 'Jordan Miller': 1641757, 'Filip Petrusev': 1630196, 'Andre Jackson Jr.': 1641748, 'Chris Livingston': 1641753, 'Colin Castleton': 1630658, "D'Moi Hodge": 1641793, 'Craig Porter Jr.': 1641854, 'Armoni Brooks': 1629717, 'Jalen Wilson': 1630592, 'Hunter Tyson': 1641816, 'Miles Bridges': 1628970, 'Jalen Slawson': 1641771, 'Javon Freeman-Liberty': 1631241, 'Keyontae Johnson': 1641749, 'Jerome Robinson': 1629010, 'Alex Fudge': 1641788, 'Nate Hinton': 1630207, 'Onuralp Bitim': 1641931, 'Sidy Cissoko': 1631321, 'Jalen Hood-Schifino': 1641720, 'Duop Reath': 1641871, 'Markquis Nowell': 1641806, 'Gui Santos': 1630611, 'Charles Bediako': 1641777, "Sir'Jabari Rice": 1641811, 'Marques Bolden': 1629716, 'Lonzo Ball': 1628366, 'Dariq Whitehead': 1641727, 'Oscar Tshiebwe': 1631131, 'Adama Sanogo': 1641766, 'Dexter Dennis': 1641926, 'Jacob Toppin': 1631210, 'Charlie Brown Jr.': 1629718, 'Jaylen Martin': 1641798, 'Jules Bernard': 1631262, 'Seth Lundy': 1641754, 'Nathan Mensah': 1641877, 'Miles Norris': 1641936, 'Javonte Smart': 1630606, 'Dmytro Skapintsev': 1631376, 'Drew Peterson': 1641809, 'Jontay Porter': 1629007, 'Brandon Williams': 1630314, 'Ricky Council IV': 1641741, 'Terquavion Smith': 1631173}