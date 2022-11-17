using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

using System.Configuration;
using System.Data.SqlClient;

namespace CRUDForm
{
    public partial class frmVIPMembersInput : Form
    {
        private string CONNECTION_STRING;
        private frmVIPMembers frm_vip_members;

        public frmVIPMembersInput(frmVIPMembers vipmembers)
        {
            InitializeComponent();

            frm_vip_members= vipmembers;
        }

        private void frmVIPMembersInput_Load(object sender, EventArgs e)
        {

        }

        private void btnOK_Click(object sender, EventArgs e)
        {
            if (txtName.Text.Trim().Length == 0)
            {
                MessageBox.Show("Please Input VIP name!",
                    "ERROR",
                    MessageBoxButtons.OK,MessageBoxIcon.Error);
                return;
            }

            CONNECTION_STRING = ConfigurationManager.AppSettings["connection_string"];
            string query = "INSERT INTO dbo.VIPmembers(member_name, member_email, member_phone) VALUES(@name, @email, @phone)";

            SqlConnection con = new SqlConnection(CONNECTION_STRING);
            SqlCommand cmd = con.CreateCommand();
            cmd.CommandText = query;
            cmd.Parameters.Add(new SqlParameter("@name",SqlDbType.VarChar,200)).Value = txtName.Text;
            cmd.Parameters.Add(new SqlParameter("@email", SqlDbType.VarChar, 100)).Value = txtEmail.Text;
            cmd.Parameters.Add(new SqlParameter("@phone", SqlDbType.VarChar, 25)).Value = txtPhone.Text;

            con.Open();
            cmd.ExecuteNonQuery();
            con.Close();

            frm_vip_members.ReloadData();

            this.Close();


        }

        private void btnCancel_Click(object sender, EventArgs e)
        {
            if (txtName.Text.Trim().Length != 0)
            {
                var buttonResult = MessageBox.Show("Close right now?",
                    "Close",
                    MessageBoxButtons.YesNo,
                    MessageBoxIcon.Warning);
                if (buttonResult == DialogResult.No)
                {
                    return;
                }
            }
            
            this.Close();
        }
    }
}
