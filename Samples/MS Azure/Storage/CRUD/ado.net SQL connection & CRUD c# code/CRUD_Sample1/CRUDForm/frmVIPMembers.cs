using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

using System.Data.SqlClient;
using System.Configuration;

namespace CRUDForm
{
    public partial class frmVIPMembers : Form
    {
        private string connection_string = "";
        private SqlConnection SqlCon = null;
        private SqlCommand SqlCmd = null;
        private SqlDataAdapter SqlApt = new SqlDataAdapter();

        private DataSet dataMain = new DataSet();


        public frmVIPMembers()
        {
            InitializeComponent();
        }

        private void btnClose_Click(object sender, EventArgs e)
        {
            this.Close();
        }

        private void frmVIPMembers_Load(object sender, EventArgs e)
        {
            connection_string = ConfigurationManager.AppSettings["connection_string"];
            ReloadData();

        }

        public void ReloadData()
        {
            dataMain.Clear();
            SqlCon = new SqlConnection(connection_string);
            string query = "SELECT * FROM dbo.VIPmembers";
            SqlCommand cmd = SqlCon.CreateCommand();
            cmd.CommandText = query;


            SqlApt.SelectCommand = cmd;
            SqlApt.Fill(dataMain);

            grdMemberList.DataSource = dataMain.Tables[0];
        }

        private void btnAdd_Click(object sender, EventArgs e)
        {
            frmVIPMembersInput vipInput = new frmVIPMembersInput(this);
            vipInput.ShowDialog();
        }
    }
}
